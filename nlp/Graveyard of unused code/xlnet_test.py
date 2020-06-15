from xlnet import xlnet
from absl.flags import FLAGS

# some code omitted here...
# initialize FLAGS
# initialize instances of tf.Tensor, including input_ids, seg_ids, and input_mask

# XLNetConfig contains hyperparameters that are specific to a model checkpoint.
xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)

# RunConfig contains hyperparameters that could be different between pretraining and finetuning.
run_config = xlnet.create_run_config(is_training=True, is_finetune=True, FLAGS=FLAGS)

# Construct an XLNet model
xlnet_model = xlnet.XLNetModel(
    xlnet_config=xlnet_config,
    run_config=run_config,
    input_ids=input_ids,
    seg_ids=seg_ids,
    input_mask=input_mask)

# Get a summary of the sequence using the last hidden state
summary = xlnet_model.get_pooled_out(summary_type="last")

# Get a sequence output
seq_out = xlnet_model.get_sequence_output()

# build your applications based on `summary` or `seq_out`