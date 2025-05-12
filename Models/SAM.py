import tensorflow as tf
from tensorflow.keras import Model
from keras.saving import register_keras_serializable

@register_keras_serializable()
class SAMModel(Model):
    def __init__(self, base_model, rho=0.05, **kwargs):
        super(SAMModel, self).__init__(**kwargs)
        self.base_model = base_model
        self.rho = rho

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def compile(self, optimizer, loss, metrics=None, **kwargs):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
        self.loss_fn = loss  # needed for custom gradient computation

    def train_step(self, data):
        x, y = data

        # First forward-backward pass
        with tf.GradientTape() as tape:
            predictions = self.base_model(x, training=True)
            loss = self.loss_fn(y, predictions)
        gradients = tape.gradient(loss, self.base_model.trainable_variables)

        # Compute perturbation
        grad_norm = tf.linalg.global_norm(gradients)
        epsilon = [g * (self.rho / (grad_norm + 1e-12)) for g in gradients]

        for var, eps in zip(self.base_model.trainable_variables, epsilon):
            var.assign_add(eps)

        # Second forward-backward pass
        with tf.GradientTape() as tape:
            predictions = self.base_model(x, training=True)
            loss = self.loss_fn(y, predictions)
        gradients = tape.gradient(loss, self.base_model.trainable_variables)

        for var, eps in zip(self.base_model.trainable_variables, epsilon):
            var.assign_sub(eps)

        self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))

        # ✅ Modern metric tracking
        self.compiled_metrics.update_state(y, predictions)
        return {m.name: m.result() for m in self.metrics}
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "rho": self.rho,
            "base_model": tf.keras.saving.serialize_keras_object(self.base_model)
        })
        return config

    @classmethod
    def from_config(cls, config):
        base_model = tf.keras.saving.deserialize_keras_object(config.pop("base_model"))
        return cls(base_model=base_model, **config)
