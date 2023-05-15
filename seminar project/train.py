import jax
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import os
from model import apply_model, update_model, clf, pretrained_bert
from jax_influence import batch_utils


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


def get_datasets(tokenizer, ds_type='train'):
    """Load MNIST train and test datasets into memory."""
    if ds_type == 'train':
        ds = pd.read_csv('mnli_train.csv')
        ds = ds.dropna()
    else:
        ds = pd.read_csv('mnli_dev.csv')
        ds = ds.dropna()
    ds['gold_label'] = ds['gold_label'].apply(lambda x: 0 if x == 'entailment' else 1)
    for i, j in enumerate(zip(ds['Sentence1'], ds['Sentence2'])):
        if i == 0:
            dummy_input = tokenizer.encode_plus(j[0], j[1],
                                                add_special_tokens=True, max_length=128,
                                                truncation=True, return_token_type_ids=True,
                                                padding="max_length", return_attention_mask=True,
                                                return_tensors="np"
                                                )
        else:
            if j[1]:
                dummy_input_append = tokenizer.encode_plus(j[0], j[1],
                                                           add_special_tokens=True, max_length=128,
                                                           truncation=True, return_token_type_ids=True,
                                                           padding="max_length", return_attention_mask=True,
                                                           return_tensors="np"
                                                           )
                for key in dummy_input.keys():
                    dummy_input[key] = np.vstack((dummy_input[key], dummy_input_append[key]))
            else:
                pass

    dummy_input['label'] = ds['gold_label'].to_numpy()
    return dummy_input


def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['label'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['label']))
    perms = perms[:steps_per_epoch * batch_size]  # Skip incomplete batch.
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []
    for perm in perms:
        batch = {'input_ids': train_ds['input_ids'][perm, ...],
                 'token_type_ids': train_ds['token_type_ids'][perm, ...],
                 'attention_mask': train_ds['attention_mask'][perm, ...],
                 'label': train_ds['label'][perm, ...]}
        batch = batch_utils.shard(batch)
        grads, loss, accuracy = jax.pmap(apply_model, axis_name='batch')(state, batch['input_ids'],
                                                                         batch['token_type_ids'],
                                                                         batch['attention_mask'],
                                                                         batch['label'])
        state = jax.pmap(update_model, axis_name='batch')(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['label'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['label']))
    perms = perms[:steps_per_epoch * batch_size]  # Skip incomplete batch.
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []
    for perm in perms:
        batch = {'input_ids': train_ds['input_ids'][perm, ...],
                 'token_type_ids': train_ds['token_type_ids'][perm, ...],
                 'attention_mask': train_ds['attention_mask'][perm, ...],
                 'label': train_ds['label'][perm, ...]}
        batch = batch_utils.shard(batch)
        grads, loss, accuracy = jax.pmap(apply_model, axis_name='batch')(state, batch['input_ids'],
                                                                         batch['token_type_ids'],
                                                                         batch['attention_mask'],
                                                                         batch['label'])
        state = jax.pmap(update_model, axis_name='batch')(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def create_train_state(rng, config):
    # randomly initialize the module
    # load the pretrained huggingface model, this is not a Flax Module
    # pretrained_bert = FlaxBertModel.from_pretrained('distilbert-base-cased')
    # create the Classifier module, extract the BERT Flax module from the huggingface model
    clf = Classifier(pretrained_bert.module)
    params = clf.init(rng, jnp.ones([1, 128]),
                      jnp.ones([1, 128]),
                      jnp.ones([1, 128]))
    # params = unfreeze(params)
    # params['params']['bert'] = pretrained_bert.params
    # params = freeze(params)
    # params = freeze(params)
    tx = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
    return train_state.TrainState.create(
        apply_fn=clf.apply, params=params, tx=tx)

def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> train_state.TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    The train state (which includes the `.params`).
  """
  # tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')
  train_ds = get_datasets(tokenizer, 'train')
  test_ds = get_datasets(tokenizer, 'test')
  rng = jax.random.PRNGKey(0)

  summary_writer = tensorboard.SummaryWriter(workdir)
  summary_writer.hparams(dict(config))

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng, config)
  state = flax.jax_utils.replicate(state)
  test_batch = {'input_ids':  test_ds['input_ids'],
                'token_type_ids': test_ds['token_type_ids'],
                'attention_mask': test_ds['attention_mask'],
                'label':  test_ds['label']}
  test_batch = batch_utils.shard(test_batch)
  for epoch in range(1, config.num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy = train_epoch(state, train_ds,
                                                    config.batch_size,
                                                    input_rng)

    _, test_loss, test_accuracy = jax.pmap(apply_model, axis_name='batch')(state,
                                                                           test_batch['input_ids'],
                                                                           test_batch['token_type_ids'],
                                                                           test_batch['attention_mask'],
                                                                           test_batch['label'])
    test_loss = np.mean(test_loss)
    test_accuracy = np.mean(test_accuracy)
    print(
        'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
        % (epoch, train_loss, train_accuracy * 100, test_loss,
           test_accuracy * 100))

    summary_writer.scalar('train_loss', train_loss, epoch)
    summary_writer.scalar('train_accuracy', train_accuracy, epoch)
    summary_writer.scalar('test_loss', test_loss, epoch)
    summary_writer.scalar('test_accuracy', test_accuracy, epoch)

  summary_writer.flush()
  return state

def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.learning_rate = 1e-4
  config.weight_decay = 1e-3
  config.batch_size = 4
  config.num_epochs = 2
  return config

# Finally, call train_and_evaluate
workdir = 'flax_mnli_fork_wd_1e-3'
final_state = train_and_evaluate(get_config(), workdir)


unrep_state = flax.jax_utils.unreplicate(final_state)
with tf.io.gfile.GFile(os.path.join(workdir, 'trained_params.flax'), 'wb') as f:
    f.write(flax.serialization.to_bytes(unrep_state.params))
