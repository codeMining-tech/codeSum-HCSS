import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import math
import random
import config
import utils




class pGraphAttentionLayer(nn.Module):
    def __init__(self,in_feature,out_feature,dropout,alpha,concat=True):
        super(pGraphAttentionLayer,self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.alpha = alpha
        self.concat = concat


        self.Wq = nn.Parameter(torch.zeros(size=(self.in_feature,self.out_feature))) # 128
        self.Wq.data.uniform_(-config.init_uniform_mag, config.init_uniform_mag)

        self.Wk = nn.Parameter(torch.zeros(size=(self.in_feature, self.out_feature)))  # 128
        self.Wk.data.uniform_(-config.init_uniform_mag, config.init_uniform_mag)

        self.Wl = nn.Parameter(torch.zeros(size=(self.in_feature, self.out_feature)))  # 128
        self.Wl.data.uniform_(-config.init_uniform_mag, config.init_uniform_mag)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.relu = nn.ReLU(self.alpha)


    def forward(self,inputs,adj):
        # inputs [32,21,512]
        q = torch.matmul(inputs,self.Wq)  # [32,21,128]
        k = torch.matmul(inputs,self.Wk)  # []

        # [32,21,21]
        s = F.softmax(torch.bmm(q,k.transpose(1,2)))

        zero_vec = -9e15 * torch.ones_like(s)
        node_num = adj.size(1)

        adj = adj+ torch.eye(node_num)
        attention = torch.where(adj > 0, s, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, inputs) # [32,21,21]   [32,21,512]
        h_prime = torch.matmul(h_prime, self.Wk)

        out = F.softmax(h_prime,dim=1)
        return out


class GRUEmbedding(nn.Module):
    def __init__(self,vocab_size):
        super(GRUEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)  # 替换 biGRu
        #self.typeEmbedding = nn.Embedding(type_num, config.embedding_dim)  # 替换 biGRu
        #self.type_linear = nn.Linear(config.embedding_dim,64)  # 一个512，一个64 其实都觉得比重有些打了，呵呵。  实现了看看。现在就是用各种方式刷指标。刷到开写论文~
        self.hidden_size = config.hidden_size

        self.GRU_layer = nn.GRU(config.embedding_dim,self.hidden_size, bidirectional=True,batch_first=True)
        #self.output_linear = nn.Linear(hidden_num, output_num)
        utils.init_rnn_wt(self.GRU_layer)
        self.hidden = None

    def forward(self,x):
        # [["decode", "stats"], ["if"], ["refill", "stats"]]

        node_len = len(x)
        result = torch.zeros(node_len, config.embedding_dim)
        for i,node in enumerate(x):
            n= torch.zeros(config.embedding_dim)

            for subtoken in node:
                s=self.embedding(torch.Tensor([subtoken]).long())
                n=n+s
            #divider = torch.full((1,512),node_len)
            #result[i][:] = torch.div(n,divider)
            result[i][:] = n/node_len

        return result

    def forward_(self,x):
        x = [torch.Tensor(n).long() for n in x]
        seq_len = [s.size(0) for s in x]
        x = pad_sequence(x, padding_value=0, batch_first=True)
        # data = pack_padded_sequence(sample, seq_len,batch_first=True,enforce_sorted=False)

        x = self.embedding(x)
        x = pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False)
        #result.append(sample)
        _, self.hidden = self.GRU_layer(x) #只取

        #out = pad_packed_sequence(self.hidden,batch_first=True,)
        out = self.hidden[0]+self.hidden[1]   # [512] 512+64 = 448

        # y = self.typeEmbedding(y)
        # y = self.type_linear(y)
        # out = torch.cat([out,y],0)
        return out


class GAT(nn.Module): # as a contrast
    def __init__(self,nfeat,nhid,nclass,dropout,alpha,nheads):
        super(GAT,self).__init__()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attentions = [pGraphAttentionLayer(nfeat,nhid,dropout=dropout,alpha=alpha,concat=True) for _ in range(nheads)]
        for i,attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i),attention)
        
        self.out_att = pGraphAttentionLayer(nhid*nheads,nclass,dropout=dropout,alpha=alpha,concat=False)

    def forward(self, x, adj):
        x = self.dropout1(x)
        x = torch.cat([att(x,adj) for att in self.attentions],dim=2)  # 节点的特征向量  在给定维度上做连接操作 第二位到底是啥呀？
        x = self.dropout2(x)
        x = F.elu(self.out_att(x,adj))
        return x

class GATEncoder(nn.Module):
    def __init__(self, vocab_size):
        super(GATEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.embedding = nn.Embedding(vocab_size,config.embedding_dim)
        self.gruembedding = GRUEmbedding(vocab_size)
        self.gat = GAT(config.nfeat,config.nhid,config.hidden_size,config.dropout,config.alpha,config.nheads) # 4头注意力
        self.gru = nn.GRU(config.embedding_dim, self.hidden_size, bidirectional=True)
        utils.init_wt_normal(self.embedding.weight)
        utils.init_rnn_wt(self.gru)

    def forward_(self, inputs: torch.Tensor, seq_lens: torch.Tensor, adjacent: torch.Tensor):

        embedded0 = self.embedding(inputs)
        embedded0 = embedded0.transpose(0, 1)  #
        embedded = torch.zeros(config.batch_size, config.max_node_length + 1, config.embedding_dim)
        _, size_em, _ = embedded0.size()

        embedded[:, :size_em, :] = embedded0
        adjacents = torch.Tensor(adjacent)
        adjacents = adjacents + adjacents.transpose(1, 2)
        # embedded = embedded.transpose(0,1)
        #print(embedded.shape)
        out = self.gat(embedded, adjacents)

        out = out.transpose(0, 1)
        output = out[:size_em, :, :]
        packed = pack_padded_sequence(output, seq_lens, enforce_sorted=False)
        outputs, hidden = self.gru(packed)
        outputs, hidden = pad_packed_sequence(outputs)  # [T, B, 2*H]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # + embedded

        return outputs, hidden  # mdify

    def forward(self, inputs: torch.Tensor, seq_lens: torch.Tensor, adjacent: torch.Tensor):

        padding = self.gruembedding([[0]])[0]
        eos = self.gruembedding([[2]])[0]
        embedded0 = torch.zeros(config.batch_size,config.max_node_length+1,512)
        for b in range(len(embedded0)):
            for t in range(len(embedded0[0])):
                embedded0[b][t][:] = padding
        for i,sample in enumerate(inputs):
            sample_v = self.gruembedding(sample)
            embedded0[i][:len(sample_v)][:] = sample_v
            embedded0[i][len(sample_v)][:] = eos  # 段末尾标志

        adjacents = torch.Tensor(adjacent)
        embedded = torch.zeros(config.batch_size,config.max_node_length+1,config.embedding_dim)
        # 16,32,512
        #print('embedding shape',embedded.shape)

        _, size_em, _ = embedded0.size()  #[B,T,I]
        
        # embedded 的size粗哦了
        embedded[:,:size_em,:] = embedded0    #只是去掉末尾的？
        adjacents = adjacents + adjacents.transpose(1,2)    # 变成对称阵？
        #embedded = embedded.transpose(0,1)
        out = self.gat(embedded,adjacents)

        out = out.transpose(0,1)
        output = out[:size_em,:,:]


        packed = pack_padded_sequence(output, seq_lens, enforce_sorted=False)
        outputs, hidden = self.gru(packed)
        #把压紧的序列再填充回来
        outputs, _ = pad_packed_sequence(outputs)  # [T, B, 2*H]  [21,32,1024]


        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] #+   加法操作  # [21,32,512]


        return outputs,hidden

#  We simply use tree-lstm network in the work "Automatic source code summarization with
# extended tree-lstm " (http://dx.doi.org/10.1109/IJCNN.2019.8851751)
class treeEncoder(nn.Module):
    def __init__(self, vocab_size):
        super(treeEncoder, self).__init__()
        pass

    def forward(self, inputs: torch.Tensor, seq_lens: torch.Tensor, adjacent: torch.Tensor):
        pass


class CODEEncoder(nn.Module):
    """
    Encoder for the code sequence(bigru)
    """

    def __init__(self, vocab_size):
        super(CODEEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_directions = 2

        # vocab_size: config.code_vocab_size for code encoder, size of sbt vocabulary for ast encoder
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)   # 这个embedding可以和那个share
        self.gru = nn.GRU(config.embedding_dim, self.hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(config.decoder_dropout_rate)
        #self.linear = nn.Linear(50 + self.hidden_size, self.hidden_size)

        #init_linear_wt(self.linear)
        self.ff = nn.Linear(config.embedding_dim, config.hidden_size)  # 这样子做效果还可以欸 一个直接给嵌入层用的，一个给GRU的输出用的。 在图计算层面，不需要考虑batch！
        self.cc = nn.Linear(config.hidden_size*2, config.hidden_size)

        utils.init_wt_normal(self.embedding.weight)
        utils.init_rnn_wt(self.gru)
        utils.init_linear_wt(self.ff)
        utils.init_linear_wt(self.cc)

    def forward(self, inputs: torch.Tensor, seq_lens: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """

        :param inputs: sorted by length in descending order, [T, B]
        :param seq_lens: should be in descending order
        :return: outputs: [T, B, H]
                hidden: [2, B, H]
        """

        embedded = self.embedding(inputs)   # [T, B, embedding_dim] # 想知道Iputs 到底是啥
        embedded = self.dropout(embedded)
        packed = pack_padded_sequence(embedded, seq_lens, enforce_sorted=False)
        outputs, hidden = self.gru(packed)
        outputs, _ = pad_packed_sequence(outputs)  # [T, B, 2*H]
        
        embedded = embedded.transpose(0,1)
        embedded = self.ff(embedded)
        embedded = embedded.transpose(0,1)
        outputs = self.cc(outputs) + embedded  # modified 8-10

        return outputs, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_directions, batch_size, self.hidden_size, device=config.device)

class Attention(nn.Module):

    def __init__(self, hidden_size=config.hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.v = nn.Parameter(torch.rand(self.hidden_size), requires_grad=True)   # [H]
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        """
        forward the net
        :param hidden: the last hidden state of encoder, [1, B, H]
        :param encoder_outputs: [T, B, H]
        :return: softmax scores, [B, 1, T]
        """
        time_step, batch_size, _ = encoder_outputs.size()  # T,32,H
        h = hidden.repeat(time_step, 1, 1).transpose(0, 1)  # [B, T, H]
        encoder_outputs = encoder_outputs.transpose(0, 1)   # [B, T, H]

        attn_energies = self.score(h, encoder_outputs)      # [B, T]
        attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)     # [B, 1, T]

        return attn_weights

    def score(self, hidden, encoder_outputs):
        """
        calculate the attention scores of each word
        :param hidden: [B, T, H]
        :param encoder_outputs: [B, T, H]
        :param coverage: [B, T]
        :return: energy: scores of each word in a batch, [B, T]
        """
        # after cat: [B, T, 2/3*H]
        # after attn: [B, T, H]
        # energy: [B, T, H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))     # [B, T, H]
        energy = energy.transpose(1, 2)     # [B, H, T]

        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)      # [B, 1, H]
        energy = torch.bmm(v, energy)   # [B, 1, T]
        return energy.squeeze(1)


class Decoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=config.hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim) # embedding的是
        self.dropout = nn.Dropout(config.decoder_dropout_rate)
        self.code_attention = Attention()
        self.gat_attention = Attention()

        self.tree_attention = Attention()

        self.gru = nn.GRU(config.embedding_dim + self.hidden_size*2, self.hidden_size) #修改
        self.out = nn.Linear(3 * self.hidden_size, config.nl_vocab_size)   #修改
        self.pick = nn.Linear(self.hidden_size, 1)
   
        if config.use_pointer_gen:
            self.p_gen_linear = nn.Linear(2*self.hidden_size+config.embedding_dim,1)


        utils.init_wt_normal(self.embedding.weight)
        utils.init_rnn_wt(self.gru)
        utils.init_linear_wt(self.out)
        utils.init_linear_wt(self.pick)

    def forward(self, inputs, last_hidden, code_outputs, gat_outputs,tree_outputs, extend_type_batch,extra_zeros ):

        """
        forward the net
        :param inputs: word input of current time step, [B]
        :param last_hidden: last decoder hidden state, [1, B, H]
        :param source_outputs: outputs of source encoder, [T, B, H]
        :param code_outputs: outputs of code encoder, [T, B, H]
        :param ast_outputs: outputs of ast encoder, [T, B, H]
        :param extend_source_batch: [B, T]
        :param extra_zeros: [B, max_oov_num]
        :return: output: [B, nl_vocab_size]
                hidden: [1, B, H]
                attn_weights: [B, 1, T]
        """
        embedded = self.embedding(inputs).unsqueeze(0)      # [1, B, embedding_dim]
        embedded = self.dropout(embedded)

        # get attn weights of source
        # calculate and add source context in order to update attn weights during training

        code_attn_weights = self.code_attention(last_hidden, code_outputs)  # [B, 1, T]
        code_context = code_attn_weights.bmm(code_outputs.transpose(0, 1))  # [B, 1, H]      [B,1,T] [B,T,H]
        code_context = code_context.transpose(0, 1)     # [1, B, H]

        gat_attn_weights = self.code_attention(last_hidden, gat_outputs)  # [B, 1, T]
        gat_context = gat_attn_weights.bmm(gat_outputs.transpose(0, 1))  # [B, 1, H]
        gat_context = gat_context.transpose(0, 1)     # [1, B, H]

        tree_attn_weights = self.code_attention(last_hidden, tree_outputs)  # [B, 1, T]
        tree_context = tree_attn_weights.bmm(tree_outputs.transpose(0, 1))  # [B, 1, H]
        tree_context = tree_context.transpose(0, 1)  # [1, B, H]

        # 多模态，加一个AST进来~
        # ast_context

        # 这里就是为了计算 二者的比例，没仔细看是怎么做的。
        poss = F.softmax(last_hidden.squeeze(0),dim=1)  #[B,H]
        poss = self.pick(poss)  #[B,1]
        poss = torch.sigmoid(poss) #[B,1]
        
        # make ratio between source code and construct is 1: 1

        context = code_context + poss*tree_context     # [1, B, H]
        context = torch.cat((gat_context,context),2)

        p_gen = None
        if config.use_pointer_gen:
            p_gen_input = torch.cat([context,last_hidden,embedded],dim=2)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)
            p_gen = p_gen.squeeze(0)
        # context 在两个地方都使用了的
        rnn_input = torch.cat([embedded, context], dim=2)   # [1, B, embedding_dim + H]
        outputs, hidden = self.gru(rnn_input, last_hidden)  # [1, B, H] for both

        outputs = outputs.squeeze(0)    # [B, H]
        context = context.squeeze(0)    # [B, H]

        vocab_dist = self.out(torch.cat([outputs, context], 1))    # [B, nl_vocab_size]
        vocab_dist = F.softmax(vocab_dist, dim=1)     # P_vocab, [B, nl_vocab_size]

        if config.use_pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            #gat_attn_weights_ = gat_attn_weights.squeeze(1)
            code_attn_weights_ = code_attn_weights.squeeze(1)
            #attn_dist = (1-p_gen)*gat_attn_weights_
            attn_dist = (1-p_gen)*code_attn_weights_

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], dim=1)
            
            final_dist = vocab_dist_.scatter_add(1,extend_type_batch,attn_dist)  # 使用code token 输入更好呀~ # 修改一哈！
            
        else:
            final_dist = vocab_dist
        
        final_dist = torch.log(final_dist + config.eps)

        return final_dist, hidden  # [32,30000]  [1,32,512]



class Model(nn.Module):

    def __init__(self, code_vocab_size, tree_vocab_size,type_vocab_size, nl_vocab_size,
                 model_file_path=None, model_state_dict=None, is_eval=False):
        super(Model, self).__init__()

        # vocabulary size for encoders
        #self.code_vocab_size = code_vocab_size
        self.is_eval = is_eval

        # init models
        # 组件是非常清晰的
        self.gat_encoder = GATEncoder(type_vocab_size)
        self.code_encoder = CODEEncoder(code_vocab_size)
        self.tree_encoder = treeEncoder(tree_vocab_size) #
        self.decoder = Decoder(nl_vocab_size)

        if config.use_cuda:
            self.gat_encoder = self.gat_encoder.cuda()
            self.tree_encoder = self.tree_encoder.cuda()
            self.code_encoder = self.code_encoder.cuda()
            self.decoder = self.decoder.cuda()

        if model_file_path:
            state = torch.load(model_file_path)
            self.set_state_dict(state)

        if model_state_dict:
            self.set_state_dict(model_state_dict)

        if is_eval:
            self.gat_encoder.eval()
            self.code_encoder.eval()
            self.tree_encoder.eval()
            self.lstmtree_encoder.eval()
            self.decoder.eval()
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')


    def forward(self, batch,adj_2,batch_size, nl_vocab, adj_1,is_test=False):
        """

        :param batch:
        :param batch_size:
        :param nl_vocab:
        :param is_test: if True, function will return before decoding
        :return: decoder_outputs: [T, B, nl_vocab_size]
        """
        code_batch, code_seq_lens,tree_batch, tree_seq_lens, \
        type_batch, type_seq_lens, nl_batch, nl_seq_lens = batch.get_regular_input()

        # encode
        # outputs: [T, B, H]
        # hidden: [2, B, H]
        # 两个encoder明显不平衡了呀


        #tree_outputs, tree_hidden = self.lstmtree_encoder(features, node_order, adjacency_list, edge_order,tree_sizes
        #tree_outputs, tree_hidden = self.tree_encoder(tree_batch, tree_seq_lens, adj_2)
        gat_outputs, graph_hidden = self.gat_encoder(type_batch, type_seq_lens, adj_1)  # 所以type batch 到底是怎么区分节点的 [T,B,H] [21,32,512]
        #tree_outputs = gat_outputs
        code_outputs, code_hidden = self.code_encoder(code_batch, code_seq_lens)    #[300,32,512]   ,hiddeb [2,32,512]  丢掉一个semantic encoder
        #code_outputs =gat_outputs
        # data for decoder
        # source_hidden = source_hidden[:1]
        code_hidden = code_hidden[:1]  # [1, B, H] #
        #code_hidden = graph_hidden[:1]  # [1, B, H]
        decoder_hidden = code_hidden  # [1, B, H]


        tree_outputs, tree_hidden = code_outputs,code_hidden
        
        if is_test:
            return code_outputs, gat_outputs,tree_outputs, decoder_hidden

        if nl_seq_lens is None:
            max_decode_step = config.max_decode_steps  # 最多输出30个词汇~
        else:
            max_decode_step = max(nl_seq_lens)

        decoder_inputs = utils.init_decoder_inputs(batch_size=batch_size, vocab=nl_vocab)  # [B] /s 时序上输入的第一个token？

        #extend_type_batch = None
        extend_code_batch = None # [32,151]
        extra_zeros = None
        if config.use_pointer_gen:  # 指针生成网络
            extend_code_batch, _, extra_zeros = batch.get_pointer_gen_input()
            #_, extend_nl_batch, extra_zeros = batch.get_pointer_gen_input()
            decoder_outputs = torch.zeros((max_decode_step, batch_size, config.nl_vocab_size+batch.max_oov_num),device=config.device)
        else:
            decoder_outputs = torch.zeros((max_decode_step, batch_size, config.nl_vocab_size), device=config.device)


        for step in range(max_decode_step):
            # decoder_outputs: [B, nl_vocab_size]
            # decoder_hidden: [1, B, H]
            # attn_weights: [B, 1, T]

            decoder_output, decoder_hidden = self.decoder(inputs=decoder_inputs,
                                             last_hidden=decoder_hidden, 
                                             code_outputs=code_outputs, 
                                             gat_outputs=gat_outputs,
                                            tree_outputs=tree_outputs,
                                             extend_type_batch=extend_code_batch,
                                             extra_zeros=extra_zeros)
            decoder_outputs[step] = decoder_output

            if config.use_teacher_forcing and random.random() < config.teacher_forcing_ratio and not self.is_eval:
                decoder_inputs = nl_batch[step]  #
            else:
                # output of last step to be the next input
                _, indices = decoder_output.topk(1)  # [B, 1]

                if config.use_pointer_gen:
                    word_indices = indices.squeeze(1).detach().cpu().numpy() # [32]
                    decoder_inputs = []
                    for index in word_indices:
                        decoder_inputs.append(utils.tune_up_decoder_input(index, nl_vocab))
                    decoder_inputs = torch.tensor(decoder_inputs, device=config.device)
                else:
                    decoder_inputs = indices.squeeze(1).detach()  # [B]
                    decoder_inputs = decoder_inputs.to(config.device)
            #config.logger.info('decode step: {}'.format(step))

        #config.logger.info('decode outputs shape: {}'.format(str(decoder_outputs.shape)))

        return decoder_outputs

    def set_state_dict(self, state_dict):
        self.code_encoder.load_state_dict(state_dict["code_encoder"])
        self.gat_encoder.load_state_dict(state_dict["gat_encoder"])
        self.tree_encoder.load_state_dict(state_dict["tree_encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
