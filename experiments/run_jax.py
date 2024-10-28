import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Tuple, Optional


class DecoderBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # Self-attention
        norm_x = nn.LayerNorm()(x)
        attention_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )(norm_x, norm_x, norm_x, deterministic=not training)
        x = x + attention_output
        
        # Feed-forward network
        norm_x = nn.LayerNorm()(x)
        ff_output = nn.Dense(self.hidden_dim * 4)(norm_x)
        ff_output = nn.gelu(ff_output)
        ff_output = nn.Dense(self.hidden_dim)(ff_output)
        ff_output = nn.Dropout(
            rate=self.dropout_rate,
            deterministic=not training
        )(ff_output)
        
        return x + ff_output

class TransformerDecoder(nn.Module):
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    max_len: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # Token embedding
        token_emb = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_dim
        )(x)
        
        # Positional embedding
        pos_emb = nn.Embed(
            num_embeddings=self.max_len,
            features=self.hidden_dim
        )(jnp.arange(x.shape[1]))
        
        x = token_emb + pos_emb
        x = nn.Dropout(
            rate=self.dropout_rate,
            deterministic=not training
        )(x)
        
        # Decoder blocks
        for _ in range(self.num_layers):
            x = DecoderBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            )(x, training)
            
        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size)(x)
        return logits

class TrainState(train_state.TrainState):
    dropout_rng: jax.random.PRNGKey

def create_train_state(
    rng: jax.random.PRNGKey,
    model: TransformerDecoder,
    learning_rate: float
) -> TrainState:
    """Creates initial training state with Adam optimizer and dropout RNG."""
    params_rng, dropout_rng = jax.random.split(rng)
    params = model.init(
        {'params': params_rng, 'dropout': dropout_rng},
        jnp.ones((1, 1), dtype=jnp.int32),
        training=False
    )
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(param_count)
    tx = optax.adamw(
        learning_rate=learning_rate,
        b1=0.9,
        b2=0.98,
        eps=1e-9,
        weight_decay=0.01
    )
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dropout_rng=dropout_rng
    )

@jax.jit
def train_step(
    state: TrainState,
    batch: jnp.ndarray,
    labels: jnp.ndarray
) -> Tuple[TrainState, jnp.ndarray]:
    """Performs a single training step."""
    dropout_rng = jax.random.fold_in(state.dropout_rng, state.step)
    
    def loss_fn(params):
        logits = state.apply_fn(
            params,
            batch,
            training=True,
            rngs={'dropout': dropout_rng}
        )
        
        pred_indices = jnp.argmax(logits, axis=-1)
            

        # Create mask for valid tokens (excluding classes 0-2 and 7)
        mask = jnp.where(labels > 2, 1.0, 0.0) * jnp.where(labels < 7, 1.0, 0.0)

        # Apply mask to predictions
        pred_indices = pred_indices * mask
        # Define the geometric transformation matrix
        transform_matrix = jnp.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.5, -0.8660254],
            [0.8660254, 0.5],
            [0.8660254, -0.5],
            [0.5, 0.8660254],
            [0.0, 0.0]
        ])
        
        pred_one_hot = jax.nn.one_hot(pred_indices, num_classes=8)
        label_one_hot = jax.nn.one_hot(labels, num_classes=8)

        pred_transformed = jnp.matmul(pred_one_hot, transform_matrix)
        label_transformed = jnp.matmul(label_one_hot, transform_matrix)
        
        pred_cumsum = jnp.cumsum(pred_transformed, axis=2)
        label_cumsum = jnp.cumsum(label_transformed, axis=2)
        x_dist =jnp.array([0])
        y_dist =jnp.array([0])
        for i in range(10):
            x_s = pred_cumsum[:,((i)*100):((i+1)*100),0].mean(1)
            x_d = label_cumsum[:,((i)*100):((i+1)*100),0].mean(1)
            x_d = jnp.abs(x_s - x_d).mean()
            x_dist = x_dist + x_d
            y_s = pred_cumsum[:,((i)*100):((i+1)*100),1].mean(1)
            y_d = label_cumsum[:,((i)*100):((i+1)*100),1].mean(1)
            y_d = jnp.abs(y_s - y_d).mean()
            y_dist = y_dist + y_d

        geom_diss = (x_dist/10) + (y_dist/10) 
        #geometric_loss = jnp.linalg.norm(pred_cumsum-label_cumsum, ord=2, axis=1).mean()
        
        return jnp.abs(geom_diss.flatten()[0])
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train_model(
    rng: jax.random.PRNGKey,
    train_data: jnp.ndarray,
    num_epochs: int = 10,
    batch_size: int = 1
):
    # Model hyperparameters
    vocab_size = 8
    hidden_dim = 1024
    num_layers = 24
    num_heads = 16
    max_len = 2048
    learning_rate = 1e-4
    
    # Initialize model and training state
    model = TransformerDecoder(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_len=max_len
    )
    state = create_train_state(rng, model, learning_rate)
    
    # Training loop
    num_batches = len(train_data) // batch_size
    for epoch in range(num_epochs):
        total_loss = 0.0
        count= 0
        minibatch = 0
        minicount = 0 
        for i in range(num_batches):
            batch_data = train_data[i * batch_size:(i + 1) * batch_size]['input_ids']
            # Shift data to create input and target sequences
            batch_input = batch_data[:, :-1]
            batch_labels = batch_data[:, 1:]
            state, loss = train_step(state, batch_input, batch_labels)
            total_loss += loss
            count+= 1
            minibatch += loss
            minicount+= 1
            if i % 10 ==0:
                print(f"Loss {minibatch.mean()/minicount}")
                minibatch = 0
                minicount = 0 
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
    
    return state
