from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.modeling import PreTrainedBertModel


class BertForLabelEncoding(PreTrainedBertModel):
    def __init__(self, config, trainable=False):
        super(BertForLabelEncoding, self).__init__(config)

        self.config = config
        self.bert = BertModel(config)
        #self.apply(self.init_bert_weights)     # don't need to perform due to pre-trained params loading

        if not trainable:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers)
        return pooled_output
