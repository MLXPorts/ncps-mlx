import mlx.core as mx
import numpy as np
from typing import Tuple

class HPC16x8:
    """
    128-bit HPC-limb integer in MLX: shape=(...,8), dtype=mx.uint16
    - Each HPC16x8 stores N HPC-limb numbers of 128 bits each.
    - Overflow saturates the top limb.
    - Negative or > 2^128-1 not supported in from_ints.
    - No shape changes; we always have 8 limbs.

    Extended with multiplication, shifting, compare, min, max as examples.
    """

    LIMB_COUNT = 8
    LIMB_BITS  = 16
    MASK       = (1 << LIMB_BITS) - 1

    def __init__(self, data: mx.array):
        if data.dtype != mx.uint16:
            raise TypeError("HPC16x8 requires dtype=mx.uint16.")
        if data.shape[-1] != self.LIMB_COUNT:
            raise ValueError(f"Last dimension must be {self.LIMB_COUNT} for 128-bit HPC.")
        self.data = data
        self.shape = data.shape[:-1]

    @classmethod
    def from_ints(cls, values, leading_shape=()):
        """
        Build HPC16x8 from an iterable of nonnegative Python ints (< 2^128 ideally).
        If it exceeds 2^128-1, we saturate to mod 2^128.

        leading_shape => if not empty, reshape final array to that shape+(8,).
        """
        host_data = []
        for val in values:
            if val < 0:
                raise ValueError("No negatives in HPC16x8")
            limbs = []
            tmp = val
            for _ in range(cls.LIMB_COUNT):
                limbs.append(tmp & cls.MASK)
                tmp >>= cls.LIMB_BITS
            host_data.append(limbs)

        host_np = np.array(host_data, dtype=np.uint16)
        mx_data = mx.array(host_np, dtype=mx.uint16)
        if leading_shape:
            mx_data = mx.reshape(mx_data, leading_shape + (cls.LIMB_COUNT,))
        return cls(mx_data)

    def to_ints(self):
        """
        CPU copy => python ints. Flatten if shape=(N,8).
        """
        arr_cpu = np.array(self.data, copy=False)
        flat = arr_cpu.reshape(-1, self.LIMB_COUNT)
        out = []
        for row in flat:
            val = 0
            shift = 0
            for limb in row:
                val |= (int(limb) << shift)
                shift += self.LIMB_BITS
            out.append(val)
        return out

    def to_float64(self):
        """
        Approximate HPC16x8 => float64. Large values => lose precision.
        """
        arr_cpu = np.array(self.data, copy=False)
        flat = arr_cpu.reshape(-1, self.LIMB_COUNT)
        out_f = np.zeros(len(flat), dtype=np.float64)
        for i, row in enumerate(flat):
            val = 0
            shift=0
            for limb in row:
                val |= (int(limb) << shift)
                shift += self.LIMB_BITS
            out_f[i] = float(val)
        return out_f.reshape(self.shape)

    # ----------------------------------------------------------------
    # Basic HPC-limb ops
    # ----------------------------------------------------------------

    def copy(self) -> "HPC16x8":
        return HPC16x8(self.data.copy())

    def add(self, other: "HPC16x8") -> "HPC16x8":
        if self.data.shape != other.data.shape:
            raise ValueError("add: shape mismatch.")
        shape = self.data.shape
        out_data = mx.zeros(shape, dtype=mx.uint16)
        carry = mx.zeros(shape[:-1], dtype=mx.uint32)

        mask_16 = mx.array([0xFFFF], dtype=mx.uint32)
        shift_16= mx.array([16],     dtype=mx.uint32)

        for i in range(self.LIMB_COUNT):
            a_32 = self.data[..., i].astype(mx.uint32)
            b_32 = other.data[..., i].astype(mx.uint32)
            s_val= mx.add(mx.add(a_32, b_32), carry)
            lo_16= mx.bitwise_and(s_val, mask_16).astype(mx.uint16)
            out_data[..., i] = lo_16
            carry = mx.right_shift(s_val, shift_16)

        # ignore final carry => saturates
        return HPC16x8(out_data)

    def sub(self, other:"HPC16x8") -> "HPC16x8":
        if self.data.shape != other.data.shape:
            raise ValueError("sub: shape mismatch.")
        shape = self.data.shape
        out_data = mx.zeros(shape, dtype=mx.uint16)
        borrow= mx.zeros(shape[:-1], dtype=mx.uint32)

        mask_16= mx.array([0xFFFF], dtype=mx.uint32)
        shift_31=mx.array([31],     dtype=mx.uint32)

        for i in range(self.LIMB_COUNT):
            a_32 = self.data[..., i].astype(mx.uint32)
            b_32 = other.data[..., i].astype(mx.uint32)
            diff_32= mx.subtract(mx.subtract(a_32, b_32), borrow)
            lo_16= mx.bitwise_and(diff_32, mask_16).astype(mx.uint16)
            out_data[..., i] = lo_16
            # borrow = signbit => diff_32<0 => top bit
            signbit = mx.right_shift(diff_32.astype(mx.int32), shift_31)
            borrow= signbit.astype(mx.uint32)

        return HPC16x8(out_data)

    # ----------------------------------------------------------------
    # Multiplication (naive O(n^2) for HPC-limb) 
    #  - We'll do a double loop in Python, but each step uses MLX ops.
    #  - For 8-limb x 8-limb => 16 partial-limb product, so we saturate final.
    # ----------------------------------------------------------------
    def mul(self, other:"HPC16x8") -> "HPC16x8":
        """
        Naive HPC-limb multiply: 
          out = sum_{i,j} ( self[i] * other[j] << (16*(j)) ), chunkwise in MLX.
        For shape=(...,8), we do this elementwise across leading dims.
        """
        if self.data.shape != other.data.shape:
            raise ValueError("mul: shape mismatch.")
        shape = self.data.shape
        out_data = mx.zeros(shape, dtype=mx.uint16)

        # We'll store partial sums in 32 bits for intermediate
        # then do carry-based addition. There's a simpler double loop approach:
        #   for i in range(8):
        #       carry=0
        #       for j in range(8):
        #         partial= out_data[..., i+j] + self.limb[i]*other.limb[j] + carry
        #         ...
        # but we only have 8 limbs total, so i+j can go up to 14. We'll saturate if i+j>=8.

        for i in range(self.LIMB_COUNT):
            # current HPC-limb for self
            a_i_32 = self.data[..., i].astype(mx.uint32)
            carry = mx.zeros(shape[:-1], dtype=mx.uint32)
            for j in range(self.LIMB_COUNT):
                # product => a_i*b_j 
                b_j_32 = other.data[..., j].astype(mx.uint32)
                prod_32= mx.multiply(a_i_32, b_j_32)   # up to 32 bits
                # add carry + current out_data[..., i+j], ignoring if i+j>=8 => saturate
                idx = i + j
                if idx< self.LIMB_COUNT:
                    old_val_32 = out_data[..., idx].astype(mx.uint32)
                    partial_sum = mx.add(mx.add(old_val_32, prod_32), carry)
                    # store new lo limb
                    lo_16= mx.bitwise_and(partial_sum, mx.array([0xFFFF], dtype=mx.uint32)).astype(mx.uint16)
                    out_data[..., idx] = lo_16
                    # new carry
                    carry= mx.right_shift(partial_sum, mx.array([16], dtype=mx.uint32))
                else:
                    # saturate => drop the result
                    # just update carry for a possible next iteration?
                    # but i+j only goes up to 14 => final might get lost
                    # We'll skip it => effectively saturating
                    break

        return HPC16x8(out_data)

    # ----------------------------------------------------------------
    # Partial bit shift: shift left up to 15 bits
    #   We'll do a single-limb approach: each step merges carry from previous limb
    # ----------------------------------------------------------------
    def shift_left_bits(self, bits:int) -> "HPC16x8":
        """
        HPC-limb left shift by 'bits' (< 16). If bits>=16, call repeated or do larger approach.
        We loop over limbs so partial bits get carried into the next. 
        Overflow saturates final limb.
        """
        if bits<=0:
            return self.copy()
        if bits>=16:
            raise ValueError("shift_left_bits: bits must be <16 in this snippet.")
        shape= self.data.shape
        out_data= mx.zeros(shape, dtype=mx.uint16)

        carry= mx.zeros(shape[:-1], dtype=mx.uint32)
        shift_32= mx.array([bits],  dtype=mx.uint32)
        mask_16= mx.array([0xFFFF], dtype=mx.uint32)

        for i in range(self.LIMB_COUNT):
            limb_32= self.data[..., i].astype(mx.uint32)
            shifted= mx.left_shift(limb_32, shift_32)
            # partial= shifted + carry
            partial= mx.add(shifted, carry)
            # out_i = partial & 0xFFFF
            lo_16= mx.bitwise_and(partial, mask_16).astype(mx.uint16)
            out_data[..., i] = lo_16
            # new carry => top bits
            carry= mx.right_shift(partial, mx.array([16], dtype=mx.uint32))

        return HPC16x8(out_data)

    def shift_right_bits(self, bits:int) -> "HPC16x8":
        """
        HPC-limb right shift by 'bits' (<16).
        We move bits from high limbs to lower. The fraction is dropped => saturate.
        """
        if bits<=0:
            return self.copy()
        if bits>=16:
            raise ValueError("shift_right_bits: bits must be <16 in this snippet.")
        shape= self.data.shape
        out_data= mx.zeros(shape, dtype=mx.uint16)

        carry= mx.zeros(shape[:-1], dtype=mx.uint32)
        shift_32= mx.array([bits], dtype=mx.uint32)
        mask_16= mx.array([0xFFFF], dtype=mx.uint32)

        # We go from top limb down to 0
        for i in reversed(range(self.LIMB_COUNT)):
            limb_32= self.data[..., i].astype(mx.uint32)
            # partial= (limb_32 <<16) + carry => SHIFT => we get out_i, new carry
            # Actually simpler approach: shift current limb right first, store in out_data,
            # then shift bits that are falling off into 'carry' for the next lower limb.

            # incorporate carry in top bits
            extended= mx.left_shift(carry, mx.array([16], dtype=mx.uint32))
            extended= mx.add(extended, limb_32)  # up to ~32 bits
            # new_limb= extended >> bits => out
            new_limb= mx.right_shift(extended, shift_32)
            lo_16= mx.bitwise_and(new_limb, mask_16).astype(mx.uint16)
            out_data[..., i] = lo_16
            # carry => the bits that got shifted out
            # carry= extended & ((1<<bits)-1) ??? 
            # Actually, if extended is 32 bits, we do 
            carry= mx.bitwise_and(extended, mx.array([(1<<bits)-1], dtype=mx.uint32))

        return HPC16x8(out_data)

    # ----------------------------------------------------------------
    # compare, min, max
    # ----------------------------------------------------------------
    def compare(self, other:"HPC16x8") -> mx.array:
        """
        Return an int32 array:  -1 if self<other, 0 if =, +1 if self>other
        shape=leading_shape, each element is the comparison result for HPC-limb
        scanning from top limb down.
        """
        if self.data.shape != other.data.shape:
            raise ValueError("compare: shape mismatch.")
        shape= self.data.shape[:-1]
        # We'll do a top-limb downward approach 
        # Because MLX doesn't have a "lex compare" built in, we do a Python loop:
        result= mx.zeros(shape, dtype=mx.int32)  # start=0 => means "tentatively equal"

        # "done_mask" => track which elements have decided < or > 
        # We'll do a fallback approach: once we find self[i]>other[i], we set result=+1 and done_mask=1
        # similarly for <.

        done_mask= mx.zeros(shape, dtype=mx.bool_)

        # from limb7 down to limb0
        for i in reversed(range(self.LIMB_COUNT)):
            s_i= self.data[..., i].astype(mx.uint32)
            o_i= other.data[..., i].astype(mx.uint32)
            # check if done_mask is false => we update result
            less_mask= mx.bitwise_and(mx.less(s_i, o_i), mx.logical_not(done_mask))
            greater_mask= mx.bitwise_and(mx.greater(s_i, o_i), mx.logical_not(done_mask))

            # update result => if less_mask => result=-1, if greater_mask => result=+1
            minus_ones= mx.full(shape, -1, dtype=mx.int32)
            plus_ones= mx.full(shape, +1, dtype=mx.int32)

            # result= where(less_mask, -1, where(greater_mask, +1, result))
            # We'll do a step at a time:
            temp_result= mx.where(less_mask, minus_ones, result)
            temp_result= mx.where(greater_mask, plus_ones, temp_result)
            result= temp_result

            # update done_mask => done or not
            newly_done= mx.bitwise_or(less_mask, greater_mask)
            done_mask= mx.bitwise_or(done_mask, newly_done)

        return result

    def min(self, other:"HPC16x8") -> "HPC16x8":
        """
        Elementwise HPC-limb min. If self<other => self, else other
        We'll use compare => result in {-1,0,1}, then pick.
        """
        comp= self.compare(other)  # shape=leading_shape
        # "self < other => result=-1 => pick self
        #  self > other => result=+1 => pick other
        #  = => pick self or other (they're the same)
        # We can do a masked approach for each limb
        shape= self.data.shape
        out_data= mx.zeros(shape, dtype=mx.uint16)

        self_mask= mx.equal(comp, mx.array([-1], dtype=mx.int32))  # if comp=-1 => pick self
        eq_mask=   mx.equal(comp, mx.array([0],  dtype=mx.int32))  # if comp=0  => same
        # for comp=+1 => pick other
        other_mask= mx.logical_not(mx.bitwise_or(self_mask, eq_mask))

        for i in range(self.LIMB_COUNT):
            s_i= self.data[..., i]
            o_i= other.data[..., i]
            # Use other_mask to select values
            pick= mx.where(other_mask, o_i, s_i)  # if other_mask => o_i, else s_i
            pick= mx.where(eq_mask, s_i, pick)    # if equal => either one (s_i)
            out_data[..., i] = pick

        return HPC16x8(out_data)

    def max(self, other:"HPC16x8") -> "HPC16x8":
        """
        Elementwise HPC-limb max => if self>other => self, else other
        """
        comp= self.compare(other)
        # self>other => comp=+1 => pick self
        # self<other => comp=-1 => pick other
        # eq => pick self
        shape= self.data.shape
        out_data= mx.zeros(shape, dtype=mx.uint16)

        self_mask= mx.equal(comp, mx.array([1], dtype=mx.int32)) 
        eq_mask=   mx.equal(comp, mx.array([0], dtype=mx.int32))

        for i in range(self.LIMB_COUNT):
            s_i= self.data[..., i]
            o_i= other.data[..., i]
            # First handle self > other case, then equal case
            pick= mx.where(self_mask, s_i, o_i)
            pick= mx.where(eq_mask, s_i, pick)
            out_data[..., i] = pick

        return HPC16x8(out_data)

    def __repr__(self):
        return f"<HPC16x8 shape={self.data.shape}, 128-bit HPC>\n"


class HPC128Double:
    """
    A 128-bit integer stored as two 64-bit limbs in MLX:
      data[..., 0] => low 64 bits
      data[..., 1] => high 64 bits
    dtype=mx.uint64 (or mx.int64 if you want signed).
    
    - shape = (*leading_shape, 2)
    - We fix shape issues for partial sums, so no broadcast mismatch.
    """

    LIMB_COUNT = 2  # [low, high]

    def __init__(self, data: mx.array):
        """
        data.shape = (..., 2)
        data.dtype in {mx.uint64, mx.int64}.
        """
        if data.shape[-1] != self.LIMB_COUNT:
            raise ValueError("Last dimension must be 2 for HPC128Double.")
        if data.dtype not in (mx.uint64, mx.int64):
            raise TypeError("HPC128Double requires int64 or uint64 dtype.")
        self.data = data
        self.shape = data.shape[:-1]  # leading shape (e.g. (3,))

    @classmethod
    def from_python_ints(cls, values, leading_shape=(), signed=False):
        """
        Build HPC128Double from a list of Python integers.
        If not signed, values must be >=0. Values beyond 2^128-1 are modded.
        
        leading_shape => e.g. (N,) => final data.shape = (N,2).
        """
        if signed:
            dtype = mx.int64
        else:
            dtype = mx.uint64
        
        # Convert Python ints -> (N,2) host array
        host_data = []
        for val in values:
            # if unsigned => treat negative as modded or raise error
            # do mod 2^128
            lo = val & 0xFFFFFFFFFFFFFFFF
            hi = (val >> 64) & 0xFFFFFFFFFFFFFFFF
            host_data.append([lo, hi])  # two 64-bit limbs

        host_np = np.array(host_data, dtype=np.uint64)  # shape=(N,2)
        mx_data = mx.array(host_np, dtype=dtype)
        if leading_shape:
            mx_data = mx.reshape(mx_data, leading_shape + (2,))
        return cls(mx_data)

    def to_python_ints(self):
        """
        Convert HPC128Double => python ints. 
        Unsigned approach => combine [lo, hi].
        If int64 => no special sign extension in this snippet.
        """
        arr_cpu = np.array(self.data, copy=False)  # shape=(...,2)
        flat = arr_cpu.reshape(-1, 2)
        out = []
        for lo, hi in flat:
            val = (int(hi) << 64) | (int(lo) & 0xFFFFFFFFFFFFFFFF)
            out.append(val)
        return out

    def copy(self) -> "HPC128Double":
        return HPC128Double(self.data.copy())

    # -----------------------------------------------------------
    # HPC-limb add => (lo1 + lo2, hi1 + hi2 + carry)
    # -----------------------------------------------------------
    def add(self, other: "HPC128Double") -> "HPC128Double":
        if self.data.shape != other.data.shape:
            raise ValueError("shape mismatch in add.")

        shape = self.data.shape
        out_data = mx.zeros(shape, dtype=self.data.dtype)

        # Separate limbs
        self_lo = self.data[..., 0].astype(mx.uint64)
        self_hi = self.data[..., 1].astype(mx.uint64)
        oth_lo  = other.data[..., 0].astype(mx.uint64)
        oth_hi  = other.data[..., 1].astype(mx.uint64)

        # lo_sum => (loA + loB) mod 2^64
        lo_sum  = mx.add(self_lo, oth_lo)  # shape=leading_shape, dtype=mx.uint64
        # carry => 1 if lo_sum< self_lo, else 0
        carry = mx.less(lo_sum, self_lo).astype(mx.uint64)

        # out_lo => lo_sum truncated to 64 bits (already done)
        out_data[..., 0] = lo_sum.astype(self.data.dtype)

        # hi_sum => hiA + hiB + carry
        hi_sum = mx.add(mx.add(self_hi, oth_hi), carry)
        out_data[..., 1] = hi_sum.astype(self.data.dtype)
        return HPC128Double(out_data)

    # -----------------------------------------------------------
    # HPC-limb sub => (lo1 - lo2, hi1 - hi2 - borrow)
    # -----------------------------------------------------------
    def sub(self, other: "HPC128Double") -> "HPC128Double":
        if self.data.shape != other.data.shape:
            raise ValueError("shape mismatch in sub.")

        shape = self.data.shape
        out_data = mx.zeros(shape, dtype=self.data.dtype)

        self_lo = self.data[..., 0].astype(mx.uint64)
        self_hi = self.data[..., 1].astype(mx.uint64)
        oth_lo  = other.data[..., 0].astype(mx.uint64)
        oth_hi  = other.data[..., 1].astype(mx.uint64)

        # lo_diff = loA - loB
        lo_diff = mx.subtract(self_lo, oth_lo)
        # borrow => 1 if loA< loB
        borrow = mx.less(self_lo, oth_lo).astype(mx.uint64)

        out_data[..., 0] = lo_diff.astype(self.data.dtype)

        # hi_diff => hiA - hiB - borrow
        hi_temp= mx.subtract(self_hi, oth_hi)
        hi_diff= mx.subtract(hi_temp, borrow)
        out_data[..., 1] = hi_diff.astype(self.data.dtype)
        return HPC128Double(out_data)

    # -----------------------------------------------------------
    # HPC-limb multiply => full 128-bit result from 64x64 => 128?
    # We'll do the "4 partial products" approach to get the low 128 bits of
    # loA+hiA * loB+hiB (real 256 bits, but we store mod 2^128).
    # -----------------------------------------------------------
    def mul(self, other: "HPC128Double") -> "HPC128Double":
        if self.data.shape != other.data.shape:
            raise ValueError("shape mismatch in mul.")

        shape = self.data.shape
        out_data = mx.zeros(shape, dtype=self.data.dtype)

        # get 64-bit limbs
        A_lo = self.data[..., 0].astype(mx.uint64)
        A_hi = self.data[..., 1].astype(mx.uint64)
        B_lo = other.data[..., 0].astype(mx.uint64)
        B_hi = other.data[..., 1].astype(mx.uint64)

        # partial products
        p1_lo, p1_hi = self._mul_64_64(A_lo, B_lo, shape[:-1])  # loA*loB
        p2_lo, p2_hi = self._mul_64_64(A_lo, B_hi, shape[:-1])  # loA*hiB
        p3_lo, p3_hi = self._mul_64_64(A_hi, B_lo, shape[:-1])  # hiA*loB
        # p4 => hiA*hiB => effectively shifts by 128 => discard => mod 2^128

        # sum => p1 + (p2 <<64) + (p3 <<64) mod 2^128
        # We'll define an "add_shift64" helper to combine (p_lo, p_hi) shifted by 64 bits.

        sum_lo = p1_lo
        sum_hi = p1_hi

        # add p2<<64
        sum_lo, sum_hi = self._add_shift64(sum_lo, sum_hi, p2_lo, shape[:-1])  
        # add p3<<64
        sum_lo, sum_hi = self._add_shift64(sum_lo, sum_hi, p3_lo, shape[:-1])  

        out_data[..., 0] = sum_lo.astype(self.data.dtype)
        out_data[..., 1] = sum_hi.astype(self.data.dtype)
        return HPC128Double(out_data)

    # ----------------------------------------------------------------
    # _mul_64_64 => produce (low64, high64) for each HPC number
    #   We'll do a 32x32 approach to detect partial overflow
    # ----------------------------------------------------------------
    def _mul_64_64(self, a64: mx.array, b64: mx.array, lead_shape) -> Tuple[mx.array, mx.array]:
        """
        Multiply two 64-bit arrays a64,b64 => produce up to 128 bits => (low64, high64).
        shape(a64) = lead_shape, same for b64.
        """
        # Convert to uint64
        a_lo32 = mx.bitwise_and(a64, mx.array([0xFFFFFFFF], dtype=mx.uint64))
        a_hi32 = mx.right_shift(a64, mx.array([32], dtype=mx.uint64))
        b_lo32 = mx.bitwise_and(b64, mx.array([0xFFFFFFFF], dtype=mx.uint64))
        b_hi32 = mx.right_shift(b64, mx.array([32], dtype=mx.uint64))

        # do partial products => up to 64 bits each
        p1 = mx.multiply(a_lo32, b_lo32)  # shape=lead_shape
        p2 = mx.multiply(a_lo32, b_hi32)
        p3 = mx.multiply(a_hi32, b_lo32)
        p4 = mx.multiply(a_hi32, b_hi32)  # might produce bits beyond 64 => we saturate

        # We'll combine them, ignoring bits beyond 128.
        # result = p1 + (p2<<32) + (p3<<32) + (p4<<64), mod 2^128
        # We'll do a small helper function for partial merges.

        lo = p1  # initial
        hi = mx.zeros(lead_shape, dtype=mx.uint64)

        # add p2<<32
        lo, hi = self._add_shift32(lo, hi, p2, 32, lead_shape)
        # add p3<<32
        lo, hi = self._add_shift32(lo, hi, p3, 32, lead_shape)
        # add p4<<64 => discard if beyond 128 => we do partial saturate
        lo, hi = self._add_shift32(lo, hi, p4, 64, lead_shape)

        return lo, hi

    def _add_shift32(self, base_lo: mx.array, base_hi: mx.array,
                     val: mx.array, shift_bits:int, lead_shape) -> Tuple[mx.array, mx.array]:
        """
        Add val<<shift_bits to the 128-bit sum (base_lo, base_hi).
        shift_bits is 32 or 64 in our usage.
        We'll do partial merges: if shift_bits=32 => we place val in 64<<32 range, etc.
        """
        # shape= lead_shape
        if shift_bits == 32:
            # parted_val => val<<32 => valLo=0..some
            # newLo = base_lo + (val <<32). Check carry
            shifted = mx.left_shift(val, mx.array([32], dtype=mx.uint64)) 
            newLo   = mx.add(base_lo, shifted)
            carry   = mx.less(newLo, base_lo).astype(mx.uint64)
            newHi   = mx.add(base_hi, carry)
            return newLo, newHi
        elif shift_bits == 64:
            # base_hi += val => ignoring carry beyond 64 bits
            newHi = mx.add(base_hi, val)
            return base_lo, newHi
        else:
            # fallback, ignoring partial bits
            return base_lo, base_hi

    def _add_shift64(self, base_lo: mx.array, base_hi: mx.array,
                     val_lo: mx.array, lead_shape) -> Tuple[mx.array, mx.array]:
        """
        A specialized 'add val<<64' => same logic: we just add 'val' to base_hi,
        ignoring final overflow beyond 128 bits.
        """
        newHi = mx.add(base_hi, val_lo)
        return base_lo, newHi

    def __repr__(self):
        return (f"<HPC128Double shape={self.data.shape}, "
                f"limbs=2 {self.data.dtype}>")


# ------------------ Demo --------------------
if __name__=="__main__":
    # Suppose we store 3 HPC numbers
    # We'll do an unsigned approach for demonstration
    A = HPC128Double.from_python_ints([1000, 65535, 2**64], leading_shape=(3,))
    B = HPC128Double.from_python_ints([1, 42, 123456], leading_shape=(3,))
    print("A =", A)
    print("B =", B)

    C = A.add(B)
    print("C = A+B =>", C)
    print("C as ints =>", C.to_python_ints())

    D = C.sub(B)
    print("D = C-B =>", D)
    print("D as ints =>", D.to_python_ints())

    M = A.mul(B)
    print("M = A*B =>", M)
    print("M as ints =>", M.to_python_ints())

# ---------------------- DEMO ----------------------
    # Basic usage
    A = HPC16x8.from_ints([1000, 65535, 2**64], leading_shape=(3,))
    B = HPC16x8.from_ints([1, 42, 123456], leading_shape=(3,))

    print("A =", A)
    print("B =", B, "\n")

    # 1) add, sub
    C = A.add(B)
    print("C = A+B =>", C)
    print("C as ints =>", C.to_ints(), "\n")

    D = C.sub(B)
    print("D = C-B =>", D)
    print("D as ints =>", D.to_ints(), "\n")

    # 2) mul
    M = A.mul(B)
    print("M = A*B =>", M)
    print("M as ints =>", M.to_ints(), "\n")

    # 3) shift left bits
    L = A.shift_left_bits(5)
    print("L = A <<5 =>", L)
    print("L as ints =>", L.to_ints(), "\n")

    # 4) compare, min, max
    comp_res= A.compare(B)
    print("compare(A,B) =>", comp_res)
    minAB = A.min(B)
    maxAB = A.max(B)
    print("min(A,B) =>", minAB, " => ", minAB.to_ints())
    print("max(A,B) =>", maxAB, " => ", maxAB.to_ints(), "\n")