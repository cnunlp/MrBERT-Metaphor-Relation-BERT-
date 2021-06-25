import logging
import math
import os
import warnings
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertConfig, PreTrainedModel

BertLayerNorm = torch.nn.LayerNorm
logger = logging.getLogger(__name__)

def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        print("it's a function named load_tf_weights_in_bert")
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model

class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class MrBERT(BertPreTrainedModel):
    def __init__(self, config, bert_model_dir, device, repr, integrate, relmodel):
        super().__init__(config)
        self.device = device
        self.repr = repr # the forms of representations(average/tag)
        self.integrate = integrate # three ways to integrate the contextual relations(concat/maxout/average)
        self.relmodel = relmodel # three ways to Model the contextual relation (linear/bilinear/neural tensor)
        self.num_labels1 = config.num_labels1 # sequence labels(binary)
        self.num_labels2 = config.num_labels2 # relation labels(binary)
        self.bert = BertModel.from_pretrained(bert_model_dir,config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sequence_classifier = nn.Linear(config.hidden_size, config.num_labels) # token classifier(sequence labeling)
        if self.relmodel == 'linear' or self.relmodel == 'nt':
            if self.integrate == 'concat':
                self.relation_linear = nn.Linear(config.hidden_size * 5, 1, bias=False) # relation vector(linear model)
            else:
                self.relation_linear = nn.Linear(config.hidden_size*2, 1, bias=False)
        if self.relmodel == 'bilinear' or self.relmodel == 'nt':
            if self.integrate == 'concat':
                self.relation_bilinear = nn.Bilinear(config.hidden_size, config.hidden_size*4, 1, bias=False) # relation matrix(bilinear/neural tensor model)
            else:
                self.relation_bilinear = nn.Bilinear(config.hidden_size, config.hidden_size,1, bias=False)
        self.init_weights()


    def slice_and_mean(self, state, begin, length):
        '''
        slice the representation and return the average representation
        '''
        vecs = torch.narrow(state, 0, begin, length)
        return torch.mean(vecs, dim=0).unsqueeze(0)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels1=None,
        labels2=None,
        embeddings=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        for i in range(len(input_ids)):
            state = sequence_output[i,:]
            cls_states = state[0, :].unsqueeze(0)
            if 1 in input_ids[i]:
                subj_indx_begin = list(input_ids[i]).index(1)
                subj_indx_end = list(input_ids[i]).index(2)
                if self.repr == 'tag':
                    subj_states = state[subj_indx_begin, :].unsqueeze(0)
                else:
                    subj_states = self.slice_and_mean(state, subj_indx_begin+1, subj_indx_end-subj_indx_begin-1)
            else:
                subj_states = torch.zeros(1, 768).to(self.device)

            verb_indx_begin = list(input_ids[i]).index(3)
            verb_indx_end = list(input_ids[i]).index(4)
            if self.repr == 'tag':
                verb_states = state[verb_indx_begin, :].unsqueeze(0)
            else:
                verb_states = self.slice_and_mean(state, verb_indx_begin+1, verb_indx_end-verb_indx_begin-1)

            if 5 in input_ids[i]:
                obj_indx_begin = list(input_ids[i]).index(5)
                obj_indx_end = list(input_ids[i]).index(6)
                if self.repr == 'tag':
                    obj_states = state[obj_indx_begin, :].unsqueeze(0)
                else:
                    obj_states = self.slice_and_mean(state, obj_indx_begin+1, obj_indx_end-obj_indx_begin-1)
            else:
                obj_states = torch.zeros(1, 768).to(self.device)
            verb_embeddings = embeddings[i]
            f = torch.nn.Sigmoid()
            if self.integrate == 'average':
                context_states = (cls_states.float() + subj_states.float() + obj_states.float() + verb_embeddings.float()) / 4
                if self.relmodel == 'linear' or self.relmodel == 'nt':
                    relation_linear_score = self.relation_linear(torch.cat((verb_states.float(), context_states), dim=-1).unsqueeze(0))
                    P = f(relation_linear_score)
                if self.relmodel == 'bilinear' or self.relmodel == 'nt':
                    relation_bilinear_score = self.relation_bilinear(verb_states.float(), context_states)
                    P = f(relation_bilinear_score)
                if self.relmodel == 'nt':
                    P = f(relation_linear_score + relation_bilinear_score)
            elif self.integrate == 'maxout':
                if self.relmodel == 'linear' or self.relmodel == 'nt':
                    relation_linear_score1 = self.relation_linear(torch.cat((verb_states.float(), cls_states.float()), dim=-1).unsqueeze(0))
                    relation_linear_score2 = self.relation_linear(torch.cat((verb_states.float(), subj_states.float()), dim=-1).unsqueeze(0))
                    relation_linear_score3 = self.relation_linear(torch.cat((verb_states.float(), obj_states.float()), dim=-1).unsqueeze(0))
                    relation_linear_score4 = self.relation_linear(torch.cat((verb_states.float(), verb_embeddings.float()), dim=-1).unsqueeze(0))
                    P = max(f(relation_linear_score1), f(relation_linear_score2), f(relation_linear_score3), f(relation_linear_score4))
                if self.relmodel == 'bilinear' or self.relmodel == 'nt':
                    relation_bilinear_score1 = self.relation_bilinear(verb_states.float(), cls_states.float())
                    relation_bilinear_score2 = self.relation_bilinear(verb_states.float(), subj_states.float())
                    relation_bilinear_score3 = self.relation_bilinear(verb_states.float(), obj_states.float())
                    relation_bilinear_score4 = self.relation_bilinear(verb_states.float(), verb_embeddings.float())
                    P = max(f(relation_bilinear_score1), f(relation_bilinear_score2), f(relation_bilinear_score3), f(relation_bilinear_score4))
                if self.relmodel == 'nt':
                    P = max(f(relation_linear_score1+relation_bilinear_score1), f(relation_linear_score2+relation_bilinear_score2), f(relation_linear_score3+relation_bilinear_score3), f(relation_linear_score4+relation_bilinear_score4))
            elif self.integrate == 'concat':
                context_states = torch.cat((subj_states.float(), obj_states.float(), cls_states.float(), verb_embeddings.float()),dim=-1)
                if self.relmodel == 'linear' or self.relmodel == 'nt':
                    relation_linear_score = self.relation_linear(torch.cat((verb_states.float(), context_states), dim=-1).unsqueeze(0))
                    P = f(relation_linear_score)
                if self.relmodel == 'bilinear' or self.relmodel == 'nt':
                    relation_bilinear_score = self.relation_bilinear(verb_states.float(), context_states)
                    P = f(relation_bilinear_score)
                if self.relmodel == 'nt':
                    P = f(relation_linear_score + relation_bilinear_score)
            logit2=P.unsqueeze(0).to(self.device)
            if i ==0:
                logits2 = logit2
            else:
                logits2 = torch.cat((logits2, logit2), 0)

        sequence_output = self.dropout(sequence_output)

        logits1 = self.sequence_classifier(sequence_output)

        outputs1 = (logits1,)
        outputs2 = (logits2,)

        if labels1 is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits1 = logits1.view(-1, self.num_labels1)
                active_labels1 = torch.where(
                    active_loss, labels1.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels1)
                )
                loss1 = loss_fct(active_logits1, active_labels1) # loss1: loss of sequence labeling
            else:
                loss1 = loss_fct(logits1.view(-1, self.num_labels1), labels1.view(-1))

            loss_function = nn.BCELoss()
            loss2 = loss_function(logits2.view(-1, 1), labels2.view(-1,1)) # loss2:loss of relation classification

            outputs1 = (loss1,) + outputs1
            outputs2 = (loss2,) + outputs2

        return outputs1, outputs2
