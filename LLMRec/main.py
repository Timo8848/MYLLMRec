from datetime import datetime
import json
import math
import os
import random
import sys
from time import time
from tqdm import tqdm
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse
from torch import autograd
import random

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utility.parser import parse_args
from Models import MM_Model, Decoder  
from utility.batch_test import *
from utility.logging import Logger
from utility.norm import build_sim, build_knn_normalized_graph

try:
    import setproctitle
except ImportError:
    setproctitle = None

args = parse_args()


def flag_enabled(flag_name):
    return bool(getattr(args, flag_name, 1))


def to_float_metric_dict(metric_dict):
    serializable = {}
    for key, value in metric_dict.items():
        if isinstance(value, np.ndarray):
            serializable[key] = [float(v) for v in value.tolist()]
        else:
            serializable[key] = float(value)
    return serializable


def normalize_user_init_embedding(raw_embeddings):
    return np.array([np.asarray(raw_embeddings[i], dtype=np.float32) for i in range(len(raw_embeddings))], dtype=np.float32)


def normalize_item_attribute_embeddings(raw_item_embeddings):
    normalized = {}
    for key, values in raw_item_embeddings.items():
        normalized[key] = np.array([np.asarray(values[i], dtype=np.float32) for i in range(len(values))], dtype=np.float32)
    return normalized


def resolve_user_profile_embedding_path(dataset_dir):
    if args.user_profile_path:
        return args.user_profile_path

    variant_candidates = {
        'pooled': ['augmented_user_init_embedding_pooled', 'augmented_user_init_embedding'],
        'history_summary': ['augmented_user_init_embedding_history_summary'],
        'structured_profile': ['augmented_user_init_embedding_structured_profile'],
    }
    for filename in variant_candidates[args.user_profile_variant]:
        candidate_path = os.path.join(dataset_dir, filename)
        if os.path.exists(candidate_path):
            return candidate_path
    raise FileNotFoundError(
        "Unable to locate prepared user profile embedding for variant '{}' under {}. "
        "Run prepare_steam_mvp.py again or pass --user_profile_path explicitly.".format(
            args.user_profile_variant,
            dataset_dir,
        )
    )


def build_run_summary(best_epoch, best_metrics):
    return {
        'experiment_name': args.experiment_name or args.title or args.dataset,
        'dataset': args.dataset,
        'seed': int(args.seed),
        'ks': [int(k) for k in eval(args.Ks)],
        'best_epoch': int(best_epoch),
        'best_metrics': to_float_metric_dict(best_metrics),
        'feature_flags': {
            'use_image_feat': flag_enabled('use_image_feat'),
            'use_text_feat': flag_enabled('use_text_feat'),
            'use_item_attribute': flag_enabled('use_item_attribute'),
            'use_user_profile': flag_enabled('use_user_profile'),
            'use_sample_augmentation': flag_enabled('use_sample_augmentation'),
        },
        'user_profile_variant': args.user_profile_variant,
        'user_profile_path': args.user_profile_path,
    }


def save_run_summary(run_summary):
    if not args.result_json_path:
        return
    result_dir = os.path.dirname(args.result_json_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    with open(args.result_json_path, 'w', encoding='utf-8') as handle:
        json.dump(run_summary, handle, indent=2)


class Trainer(object):
    def __init__(self, data_config):
       
        experiment_label = args.experiment_name or args.title or args.dataset
        self.task_name = "%s_%s_%s_seed%s" % (
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            experiment_label,
            args.cf_model,
            args.seed,
        )
        self.logger = Logger(filename=self.task_name, is_debug=args.debug)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        self.mess_dropout = eval(args.mess_dropout)
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.use_image_feat = flag_enabled('use_image_feat')
        self.use_text_feat = flag_enabled('use_text_feat')
        self.use_item_attribute = flag_enabled('use_item_attribute')
        self.use_user_profile = flag_enabled('use_user_profile')
        self.use_sample_augmentation = flag_enabled('use_sample_augmentation')
 
        self.image_feats = np.load(args.data_path + '{}/image_feat.npy'.format(args.dataset))
        self.text_feats = np.load(args.data_path + '{}/text_feat.npy'.format(args.dataset))
        self.image_feat_dim = self.image_feats.shape[-1]
        self.text_feat_dim = self.text_feats.shape[-1]

        dataset_dir = args.data_path + args.dataset + '/'
        self.ui_graph = self.ui_graph_raw = pickle.load(open(dataset_dir + 'train_mat','rb'))
        # get user embedding  
        user_profile_path = resolve_user_profile_embedding_path(dataset_dir)
        args.user_profile_path = user_profile_path
        augmented_user_init_embedding = pickle.load(open(user_profile_path,'rb'))
        self.user_init_embedding = normalize_user_init_embedding(augmented_user_init_embedding)
        # get separate embedding matrix 
        augmented_atttribute_embedding_dict = pickle.load(open(dataset_dir + 'augmented_atttribute_embedding_dict','rb'))
        self.item_attribute_embedding = normalize_item_attribute_embeddings(augmented_atttribute_embedding_dict)
        augmented_sample_path = dataset_dir + 'augmented_sample_dict'
        self.augmented_sample_dict = pickle.load(open(augmented_sample_path,'rb')) if os.path.exists(augmented_sample_path) else {}
        self.logger.logging("user_profile_variant: %s" % args.user_profile_variant)
        self.logger.logging("user_profile_path: %s" % user_profile_path)

        self.image_ui_index = {'x':[], 'y':[]}
        self.text_ui_index = {'x':[], 'y':[]}

        self.n_users = self.ui_graph.shape[0]
        self.n_items = self.ui_graph.shape[1]        
        self.iu_graph = self.ui_graph.T
  
        self.ui_graph = self.csr_norm(self.ui_graph, mean_flag=True)
        self.iu_graph = self.csr_norm(self.iu_graph, mean_flag=True)
        self.ui_graph = self.matrix_to_tensor(self.ui_graph)
        self.iu_graph = self.matrix_to_tensor(self.iu_graph)
        self.image_ui_graph = self.text_ui_graph = self.ui_graph
        self.image_iu_graph = self.text_iu_graph = self.iu_graph

        self.model_mm = MM_Model(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout, self.image_feats, self.text_feats, self.user_init_embedding, self.item_attribute_embedding)
        self.model_mm = self.model_mm.to(device)
        self.decoder = Decoder(self.user_init_embedding.shape[1]).to(device)


        self.optimizer = optim.AdamW(
        [
            {'params':self.model_mm.parameters()},      
        ]
            , lr=self.lr)  
        
        self.de_optimizer = optim.AdamW(
        [
            {'params':self.decoder.parameters()},      
        ]
            , lr=args.de_lr)  



    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum+1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)
        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum+1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)
        if mean_flag == False:
            return rowsum_diag*csr_mat*colsum_diag
        else:
            return rowsum_diag*csr_mat

    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  #
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #
        values = torch.from_numpy(cur_matrix.data)  #
        shape = torch.Size(cur_matrix.shape)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32, device=device)

    def innerProduct(self, u_pos, i_pos, u_neg, j_neg):  
        pred_i = torch.sum(torch.mul(u_pos,i_pos), dim=-1) 
        pred_j = torch.sum(torch.mul(u_neg,j_neg), dim=-1)  
        return pred_i, pred_j

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

    def sim(self, z1, z2):
        z1 = F.normalize(z1)  
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def feat_reg_loss_calculation(self, g_item_image, g_item_text, g_user_image, g_user_text):
        feat_reg = torch.tensor(0.0, dtype=torch.float32, device=device)
        if self.use_image_feat:
            feat_reg = feat_reg + 1./2*(g_item_image**2).sum() + 1./2*(g_user_image**2).sum()
        if self.use_text_feat:
            feat_reg = feat_reg + 1./2*(g_item_text**2).sum() + 1./2*(g_user_text**2).sum()
        feat_reg = feat_reg / max(1, self.n_items)
        return args.feat_reg_decay * feat_reg

    def prune_loss(self, pred, drop_rate):
        ind_sorted = torch.argsort(pred.detach())
        loss_sorted = pred[ind_sorted]
        remember_rate = 1 - drop_rate
        num_remember = int(remember_rate * len(loss_sorted))
        ind_update = ind_sorted[:num_remember]
        loss_update = pred[ind_update]
        return loss_update.mean()

    def mse_criterion(self, x, y, alpha=3):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        tmp_loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        tmp_loss = tmp_loss.mean()
        loss = F.mse_loss(x, y)
        return loss

    def sce_criterion(self, x, y, alpha=1):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        loss = (1-(x*y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean() 
        return loss

    def test(self, users_to_test, is_val):
        self.model_mm.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings, *rest = self.model_mm(self.ui_graph, self.iu_graph, self.image_ui_graph, self.image_iu_graph, self.text_ui_graph, self.text_iu_graph)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)
        return result

    def train(self):

        now_time = datetime.now()
        run_time = datetime.strftime(now_time,'%Y_%m_%d__%H_%M_%S')

        training_time_list = []
        stopping_step = 0
        test_ret = {'recall': np.zeros(len(eval(args.Ks))), 'precision': np.zeros(len(eval(args.Ks))), 'ndcg': np.zeros(len(eval(args.Ks))), 'hit_ratio': np.zeros(len(eval(args.Ks))), 'auc': 0.}
        best_metrics = copy.deepcopy(test_ret)
        best_epoch = -1

        n_batch = data_generator.n_train // args.batch_size + 1
        best_recall = -np.inf
        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            contrastive_loss = 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            sample_time = 0.
            build_item_graph = True

            self.gene_u, self.gene_real, self.gene_fake = None, None, {}
            self.topk_p_dict, self.topk_id_dict = {}, {}

            for idx in tqdm(range(n_batch)):
                self.model_mm.train()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()

                # augment samples
                if self.use_sample_augmentation and args.aug_sample_rate > 0:
                    candidate_users_aug = [user for user in users if user in self.augmented_sample_dict]
                    users_aug = random.sample(candidate_users_aug, int(len(users) * args.aug_sample_rate)) if candidate_users_aug else []
                    pos_items_aug = [self.augmented_sample_dict[user][0] for user in users_aug if (self.augmented_sample_dict[user][0] < self.n_items and self.augmented_sample_dict[user][1] < self.n_items)]
                    neg_items_aug = [self.augmented_sample_dict[user][1] for user in users_aug if (self.augmented_sample_dict[user][0] < self.n_items and self.augmented_sample_dict[user][1] < self.n_items)]
                    users_aug = [user for user in users_aug if (self.augmented_sample_dict[user][0] < self.n_items and self.augmented_sample_dict[user][1] < self.n_items)]
                    self.new_batch_size = len(users_aug)
                    users += users_aug
                    pos_items += pos_items_aug
                    neg_items += neg_items_aug


                sample_time += time() - sample_t1       
                user_presentation_h, item_presentation_h, image_i_feat, text_i_feat, image_u_feat, text_u_feat \
                                , user_prof_feat_pre, item_prof_feat_pre, user_prof_feat, item_prof_feat, user_att_feats, item_att_feats, i_mask_nodes, u_mask_nodes \
                        = self.model_mm(self.ui_graph, self.iu_graph, self.image_ui_graph, self.image_iu_graph, self.text_ui_graph, self.text_iu_graph)
                
                u_bpr_emb = user_presentation_h[users]
                i_bpr_pos_emb = item_presentation_h[pos_items]
                i_bpr_neg_emb = item_presentation_h[neg_items]
                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_bpr_emb, i_bpr_pos_emb, i_bpr_neg_emb)
       
                # modal feat
                mm_mf_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                if self.use_image_feat:
                    image_u_bpr_emb = image_u_feat[users]
                    image_i_bpr_pos_emb = image_i_feat[pos_items]
                    image_i_bpr_neg_emb = image_i_feat[neg_items]
                    image_batch_mf_loss, _, _ = self.bpr_loss(image_u_bpr_emb, image_i_bpr_pos_emb, image_i_bpr_neg_emb)
                    mm_mf_loss = mm_mf_loss + image_batch_mf_loss
                if self.use_text_feat:
                    text_u_bpr_emb = text_u_feat[users]
                    text_i_bpr_pos_emb = text_i_feat[pos_items]
                    text_i_bpr_neg_emb = text_i_feat[neg_items]
                    text_batch_mf_loss, _, _ = self.bpr_loss(text_u_bpr_emb, text_i_bpr_pos_emb, text_i_bpr_neg_emb)
                    mm_mf_loss = mm_mf_loss + text_batch_mf_loss

                batch_mf_loss_aug = torch.tensor(0.0, dtype=torch.float32, device=device)
                if self.use_user_profile and self.use_item_attribute:
                    for value in item_att_feats:
                        u_g_embeddings_aug = user_prof_feat[users]
                        pos_i_g_embeddings_aug = item_att_feats[value][pos_items]
                        neg_i_g_embeddings_aug = item_att_feats[value][neg_items]
                        tmp_batch_mf_loss_aug, _, _ = self.bpr_loss(u_g_embeddings_aug, pos_i_g_embeddings_aug, neg_i_g_embeddings_aug)
                        batch_mf_loss_aug = batch_mf_loss_aug + tmp_batch_mf_loss_aug

                feat_emb_loss = self.feat_reg_loss_calculation(image_i_feat, text_i_feat, image_u_feat, text_u_feat)

                att_re_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                if args.mask and (self.use_user_profile or self.use_item_attribute):
                    input_i = {} 
                    if self.use_item_attribute and i_mask_nodes is not None and len(i_mask_nodes) > 0:
                        for value in item_att_feats.keys():
                            input_i[value] = item_att_feats[value][i_mask_nodes]
                    masked_user_prof_feat = user_prof_feat[u_mask_nodes] if (self.use_user_profile and u_mask_nodes is not None and len(u_mask_nodes) > 0) else torch.zeros((0, user_prof_feat.shape[1]), dtype=torch.float32, device=device)
                    decoded_u, decoded_i = self.decoder(masked_user_prof_feat, input_i)
                    if args.feat_loss_type == 'mse':
                        if self.use_user_profile and u_mask_nodes is not None and len(u_mask_nodes) > 0:
                            att_re_loss = att_re_loss + self.mse_criterion(decoded_u, torch.tensor(self.user_init_embedding[u_mask_nodes]).to(device), alpha=args.alpha_l)
                        if self.use_item_attribute and decoded_i is not None and i_mask_nodes is not None and len(i_mask_nodes) > 0:
                            for index, value in enumerate(item_att_feats.keys()):
                                att_re_loss = att_re_loss + self.mse_criterion(decoded_i[index], torch.tensor(self.item_attribute_embedding[value][i_mask_nodes]).to(device), alpha=args.alpha_l)
                    elif args.feat_loss_type == 'sce':
                        if self.use_user_profile and u_mask_nodes is not None and len(u_mask_nodes) > 0:
                            att_re_loss = att_re_loss + self.sce_criterion(decoded_u, torch.tensor(self.user_init_embedding[u_mask_nodes]).to(device), alpha=args.alpha_l)
                        if self.use_item_attribute and decoded_i is not None and i_mask_nodes is not None and len(i_mask_nodes) > 0:
                            for index, value in enumerate(item_att_feats.keys()):
                                att_re_loss = att_re_loss + self.sce_criterion(decoded_i[index], torch.tensor(self.item_attribute_embedding[value][i_mask_nodes]).to(device), alpha=args.alpha_l)

                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss + feat_emb_loss + args.aug_mf_rate*batch_mf_loss_aug + args.mm_mf_rate*mm_mf_loss + args.att_re_rate*att_re_loss
                nn.utils.clip_grad_norm_(self.model_mm.parameters(), max_norm=1.0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      #+ ssl_loss2 #+ batch_contrastive_loss
                self.optimizer.zero_grad()  
                batch_loss.backward(retain_graph=False)
                
                self.optimizer.step()

                loss += float(batch_loss.detach())
                mf_loss += float(batch_mf_loss.detach())
                emb_loss += float(batch_emb_loss.detach())
                reg_loss += float(batch_reg_loss)
    
            del user_presentation_h, item_presentation_h, u_bpr_emb, i_bpr_neg_emb, i_bpr_pos_emb

            if math.isnan(loss) == True:
                self.logger.logging('ERROR: loss is nan.')
                sys.exit()

            if (epoch + 1) % args.verbose != 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f  + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss, contrastive_loss)
                training_time_list.append(time() - t1)
                self.logger.logging(perf_str)

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            ret = self.test(users_to_test, is_val=False)  #^-^
            training_time_list.append(t2 - t1)

            t3 = time()

            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f, %.5f, %.5f], ' \
                           'precision=[%.5f, %.5f, %.5f, %.5f], hit=[%.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0], ret['recall'][1], ret['recall'][2],
                            ret['recall'][-1],
                            ret['precision'][0], ret['precision'][1], ret['precision'][2], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][1], ret['hit_ratio'][2], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], ret['ndcg'][-1])
                self.logger.logging(perf_str)

            if ret['recall'][1] > best_recall:
                best_recall = ret['recall'][1]
                test_ret = self.test(users_to_test, is_val=False)
                best_metrics = copy.deepcopy(test_ret)
                best_epoch = epoch
                self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[1], test_ret['recall'][1], test_ret['precision'][1], test_ret['ndcg'][1]))
                stopping_step = 0
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                self.logger.logging('#####Early stopping steps: %d #####' % stopping_step)
            else:
                self.logger.logging('#####Early stop! #####')
                break
        self.logger.logging(str(test_ret))
        run_summary = build_run_summary(best_epoch=best_epoch, best_metrics=best_metrics)
        self.logger.logging("RunSummary: %s" % json.dumps(run_summary))

        return run_summary


    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./(2*(users**2).sum()+1e-8) + 1./(2*(pos_items**2).sum()+1e-8) + 1./(2*(neg_items**2).sum()+1e-8)        
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores+1e-8)
        mf_loss = - self.prune_loss(maxi, args.prune_loss_drop_rate)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss
    

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    set_seed(args.seed)
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    trainer = Trainer(data_config=config)
    run_summary = trainer.train()
    save_run_summary(run_summary)
    print(json.dumps(run_summary))
