from ncps.mini_keras.api_export import keras_mini_export

@keras_mini_export("ncps.mini_keras.saving.KerasSaveableBase")
class KerasSaveableBase:
    """Base class for objects that can be saved and loaded."""
    
    def save_own_variables(self, store):
        """Save the object's variables."""
        pass

    def load_own_variables(self, store):
        """Load the object's variables."""
        pass
        
    def get_config(self):
        """Return configuration of the saveable object."""
        return {}

    def save_assets(self, store):
        """Save the object's assets."""
        pass

    def load_assets(self, store):
        """Load the object's assets."""
        pass
