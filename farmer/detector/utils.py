"""Some utils for SSD."""

import numpy as np
import tensorflow as tf


class BBoxUtility(object):
    """Utility class to do some stuff with bounding boxes and priors.

    # Arguments
        num_classes: Number of classes including background.
        priors: Priors and variances, numpy tensor of shape (num_priors, 8),
                priors[i] = [xmin, ymin, xmax, ymax, varxc, varyc, varw, varh].
        overlap_threshold: Threshold to assign box to a prior.
        nms_thresh: Nms threshold. # Nms：Non-Maximum Suppression, threshhold：閾値
        top_k: Number of total bboxes to be kept per image after nms step.

    # References
        https://arxiv.org/abs/1512.02325
    """
    # TODO add setter methods for nms_thresh and top_K
    def __init__(self, num_classes, priors=None, overlap_threshold=0.5,
                 nms_thresh=0.45, top_k=400):
        self.num_classes = num_classes
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors) # bbox_util = BBoxUtility(NUM_CLASSES, priors) → len(priors)=7308
        self.overlap_threshold = overlap_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k
        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.scores = tf.placeholder(dtype='float32', shape=(None,))
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

    # property関数について（参考：https://tmg0525.hatenadiary.jp/entry/2017/10/08/205517）
    # name = property(get_name, set_name)
    # @property：getterのメソッドにつける
    # @プロパティ名.setter：setterのメソッドにつける。例）name.setter

    @property # getter
    def nms_thresh(self):
        return self._nms_thresh # 先頭に__で非公開属性

    @nms_thresh.setter # setter
    def nms_thresh(self, value):
        self._nms_thresh = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)

    @property # getter
    def top_k(self):
        return self._top_k

    @top_k.setter # setter
    def top_k(self, value):
        self._top_k = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)

    def iou(self, box):
        """Compute intersection over union for the box with all priors.

        ・GTボックスを全てのデフォルトボックスと比較する
        ・type(priors) = numpy.ndarray
        ・priors.shape = (7308, 8)
        ・Priors and variances, numpy tensor of shape (num_priors, 8)
        ・priors[i] = [xmin, ymin, xmax, ymax, varxc, varyc, varw, varh]

        # Arguments
            box: Box, numpy tensor of shape (4,).

        # Return
            iou: Intersection over union, numpy tensor of shape (num_priors).
        """
        # compute intersection
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])
        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # compute union
        area_pred = (box[2] - box[0]) * (box[3] - box[1])
        area_gt = (self.priors[:, 2] - self.priors[:, 0])
        area_gt *= (self.priors[:, 3] - self.priors[:, 1])
        union = area_pred + area_gt - inter
        # compute iou
        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True):
        """Encode box for training, do it only for assigned priors.

        ・GTボックスが適合したデフォルトボックスに変換される

        # Arguments
            box: Box, numpy tensor of shape (4,).
            return_iou: Whether to concat iou to encoded values.

        # Return
            encoded_box: Tensor with encoded box
                numpy tensor of shape (num_priors, 4 + int(return_iou)).
        """
        # iou
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_priors, 4 + return_iou)) # .shape = (7308, 5)
        assign_mask = iou > self.overlap_threshold # iou > 0.5 のブーリアン配列
        # iou > 0.5を満たすボックスが最低1つある状態を作る
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True # iouの最大値をTrueにする
        # 最後の要素をiouの値にする
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        # iouの閾値を超えたデフォルトボックス
        assigned_priors = self.priors[assign_mask]
        # GTボックスの中心座標, 幅高さ
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        # 適合したデフォルトボックスの中心座標, 幅高さ
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])
        # we encode variance
        # 原著論文：Training objectiveの処理
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center # GTボックス - 適合デフォルトボックス
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh # 正規化なのか？
        encoded_box[:, :2][assign_mask] /= assigned_priors[:, -4:-2] # varxc, varyc
        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] /= assigned_priors[:, -2:] # varw, varh
        return encoded_box.ravel() # 1次元配列に変更、速いらしい

    def assign_boxes(self, boxes):
        """Assign boxes to priors for training.

        ・GTボックスをデフォルトボックスに当てはめる

        # Arguments
            boxes: Box, numpy tensor of shape (num_boxes, 4 + num_classes),
                num_classes without background.

        # Return
            assignment: Tensor with assigned boxes,
                numpy tensor of shape (num_boxes, 4 + num_classes + 8),
                priors in ground truth are fictitious,
                assignment[:, -8] has 1 if prior should be penalized
                    or in other words is assigned to some ground truth box,
                assignment[:, -7:] are all 0. See loss for more details.
        """
        # 教師データ
        assignment = np.zeros((self.num_priors, 4 + self.num_classes + 8)) # (7308, 33) → predictで出てくるやつだから, これがラベル
        assignment[:, 4] = 1.0 # backgroundカテゴリを1
        # GTボックスがなければ背景を検出する
        if len(boxes) == 0:
            return assignment
        # GTボックスをエンコードする
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4]) # GTボックスの行に関数を適用する
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5) # 1次元配列を元に戻す→(1, 7308, 5)
        # 最も適合したiou値を抽出
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0 # True
        best_iou_idx = best_iou_idx[best_iou_mask] # インデックスのarrayができる
        assign_num = len(best_iou_idx) # 1だよな
        encoded_boxes = encoded_boxes[:, best_iou_mask, :] # 次元が増える(1, 1, 7308, 5)?
        # assignmentの4つに座標
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,
                                                         np.arange(assign_num),
                                                         :4] # encodeされた座標
        # カテゴリを全て0に
        assignment[:, 4][best_iou_mask] = 0 # GTボックスがある状態だから、backgroundカテゴリは用済み
        # クラスを格納
        assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_idx, 4:]
        # priorboxの先頭が1
        assignment[:, -8][best_iou_mask] = 1
        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox, variances):
        """Convert bboxes from local predictions to shifted priors. 画像でのリアルな位置に変換する（正規化はされている）

        # Arguments
            mbox_loc: Numpy array of predicted locations. # (7308, 4) → 物体がここにあるっていう位置
            mbox_priorbox: Numpy array of prior boxes. → default boxes? # (7308, 4) → デフォルトボックスの位置
            variances: Numpy array of variances. # (7308, 4) → デフォルトボックスとの差異？

            ・そのデフォルトボックスが物体からどれだけ離れているかがpredictの結果 → mbox_loc = delta
            ・デフォルトボックスの座標を基準としてmbox_loc分ズラす
            ・https://avinton.com/blog/2018/03/single-shot-multibox-detector-explained1/
            ・デフォルトボックスとGTボックスの関係がpriors
            ・mbox_loc(predict) - mbox_priorbox(default = ground truth) - variances()

        # Return
            decode_bbox: Shifted priors.
        """
        # 処理は画像ごとに行われるので、(7308, 4)
        # priorboxの処理
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0] # デフォルトボックスの幅
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1] # デフォルトボックスの高さ
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0]) # デフォルトボックスの幅中心
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1]) # デフォルトボックスの高さ中心
        # locの処理
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width * variances[:, 0] # + prior_center_x
        decode_bbox_center_x += prior_center_x
        
        decode_bbox_center_y = mbox_loc[:, 1] * prior_width * variances[:, 1] # prior_heightじゃない？
        decode_bbox_center_y += prior_center_y
        
        decode_bbox_width = np.exp(mbox_loc[:, 2] * variances[:, 2])
        decode_bbox_width *= prior_width
        
        decode_bbox_height = np.exp(mbox_loc[:, 3] * variances[:, 3])
        decode_bbox_height *= prior_height

        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0) # 正規化してあるので1を超えないための処理
        return decode_bbox

    def detection_out(self, predictions, background_label_id=0, keep_top_k=200,
                      confidence_threshold=0.01):
        """Do non maximum suppression (nms) on prediction results. + decode boxes

        # Arguments
            predictions: Numpy array of predicted values.
            num_classes: Number of classes for prediction.
            background_label_id: Label of background class.
            keep_top_k: Number of total bboxes to be kept per image after nms step.
            confidence_threshold: Only consider detections, whose confidences are larger than a threshold.

        # Return
            results: List of predictions for every picture. Each prediction is: [label, confidence, xmin, ymin, xmax, ymax]
        """
        
        # predictions[0][0] = [4loc, 21conf, 4priorbox, 4variances]
        mbox_loc = predictions[:, :, :4] # (4, 7308, 4)
        variances = predictions[:, :, -4:] # (4, 7308, 4)
        mbox_priorbox = predictions[:, :, -8:-4] # (4, 7308, 4)
        mbox_conf = predictions[:, :, 4:-8] # (4, 7308, 21)
        results = []

        # 画像ごとに処理
        for i in range(len(mbox_loc)):
            results.append([])
            decode_bbox = self.decode_boxes(mbox_loc[i], mbox_priorbox[i], variances[i])
            # クラスごとに処理
            for c in range(self.num_classes): # self.num_classes=21
                # バックグラウンドクラスはパス
                if c == background_label_id: # background_label_id=0
                    continue
                # そのクラスのconfidenceを取得
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > confidence_threshold # False or True
                # そのクラスが検出されたかどうか
                if len(c_confs[c_confs_m]) > 0: # c_confsのなかでconfidence_threshholdを超えるもののarray、要は各クラスが検出されたかどうか
                    boxes_to_process = decode_bbox[c_confs_m] # decodeされたbox
                    confs_to_process = c_confs[c_confs_m] # 閾値を超えているconf
                    feed_dict = {self.boxes: boxes_to_process,
                                 self.scores: confs_to_process}
                    idx = self.sess.run(self.nms, feed_dict=feed_dict) # nmsされた結果のindexを格納
                    good_boxes = boxes_to_process[idx] # 要はベストなboxたち
                    confs = confs_to_process[idx][:, None] # そのベストなboxのconf
                    labels = c * np.ones((len(idx), 1)) # まとめてクラスナンバーを作成
                    c_pred = np.concatenate((labels, confs, good_boxes), axis=1) # 結合
                    results[-1].extend(c_pred) # resultsの右に足していく
            # データが存在したときに処理
            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1]) # arrayにする
                argsort = np.argsort(results[-1][:, 1])[::-1] # confでソートする？
                results[-1] = results[-1][argsort] # ソートしたもので上書き
                results[-1] = results[-1][:keep_top_k] # keep_top_kまでで打ち切り
        # return [label, confidence, xmin, ymin, xmax, ymax]
        return results
