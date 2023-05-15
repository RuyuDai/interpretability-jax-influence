import jax
from flax import linen as nn
from transformers import AutoTokenizer, FlaxBertModel, BertConfig
import jax.numpy as jnp


# see Hans(2021) for retrieving the pretrained model: https://github.com/xhan77/influence-function-analysis
pretrained_model = '/content/drive/MyDrive/Interpretability/influence-function-analysis-master/NLI_tagger_output_bert_e3/pytorch_model.bin'
config = BertConfig.from_json_file("/content/drive/MyDrive/Interpretability/influence-function-analysis-master/NLI_tagger_output_bert_e3/bert_config.json")

class Classifier(nn.Module):
    bert: nn.Module

    def setup(self):
        self.fc = nn.Dense(features=2)

    def __call__(self, input_ids, token_type_ids, attention_mask):
        out = self.bert(input_ids, token_type_ids, attention_mask)
        out = out.pooler_output
        out = self.fc(out)
        return out

# load the pretrained huggingface model, this is not a Flax Module
pretrained_bert = FlaxBertModel.from_pretrained(pretrained_model_name_or_path=pretrained_model,
                                                from_pt=True, config=config)
# create the Classifier module, extract the BERT Flax module from the huggingface model
clf = Classifier(pretrained_bert.module)

def cross_entropy_loss(logits, labels):
    return -jnp.mean(jnp.sum(labels * logits, axis=-1))

def compute_metrics(logits, labels):
    one_hot = jax.nn.one_hot(labels, 2)
    loss = cross_entropy_loss(logits, one_hot)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics

def apply_model(state, input_ids, token_type_ids, attention_mask, labels):
    """Computes gradients, loss and accuracy for a single batch."""
    one_hot = jax.nn.one_hot(labels, 2)
    def loss_fn(params):
        logits = clf.apply(params, input_ids, token_type_ids, attention_mask)
        loss = cross_entropy_loss(logits=logits, labels=one_hot)
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, 'batch')
    metrics = compute_metrics(logits, labels)
    metrics = jax.tree_map(lambda x: jax.lax.pmean(x, 'batch'), metrics)
    return grads, metrics['loss'], metrics['accuracy']

def update_model(state, grads):
    return state.apply_gradients(grads=grads)
