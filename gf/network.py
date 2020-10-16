import numpy as np
import os

# Suppressing Caffe Warnings
# os.environ['GLOG_minloglevel'] = '3'
# import caffe
import cv2


from .data import GFImage



BasePath = os.path.dirname(os.path.dirname(__file__))
DefaultProto = os.path.join(BasePath, "utils", "deploy_demo.prototxt")
DefaultModel = os.path.join(BasePath, "utils", "binary_w.caffemodel")


output_layer_names = ["fc_0_0", "fc_1_0", "fc_m1_0", "fc_0_1", "fc_0_m1"]

class GFNetwork:

    def __init__(self, proto=DefaultProto, model=DefaultModel):
        self.model = cv2.dnn.readNetFromCaffe(proto, model)

    def forward(self, source, eye_index=0):
        """
        source should be an instance of GFImage from file gf/data.py
        """
        self.source = source
        if len(source.annotations) <= eye_index:
            raise ValueError("The position of the eyes is not known")

        ## Caffe syntax
        # self.model.blobs['data'].data[...] = source.getFormattedImage()
        # self.model.blobs['face'].data[...] = source.getFormattedHeadImage()
        # self.model.blobs['eyes_grid'].data[...] =  source.getMask()


        ## Opencv equivalent
        blob_full_img = cv2.dnn.blobFromImage(source.getFormattedImage().transpose(1,2,0),
                                              swapRB=True)
        blob_head_img = cv2.dnn.blobFromImage(source.getFormattedHeadImage().transpose(1,2,0),
                                              swapRB=True)
        blob_mask = cv2.dnn.blobFromImage(source.getMask().astype(np.float32))
        self.model.setInput(blob_full_img, name = "data")
        self.model.setInput(blob_head_img, name = "face")
        self.model.setInput(blob_mask, name = "eyes_grid")
        self.result = self.model.forwardAndRetrieve(output_layer_names)


        self._compute_heatmap()
        self._compute_gaze()


    # Internal Methods
    def _format_grid(self, name):
        grid = self.result[output_layer_names.index(name)][0].reshape((5,5))
        grid = np.exp(0.3 * grid)
        grid = grid/np.sum(grid)
        return grid

    def _compute_heatmap(self):
        f_0_0 = self._format_grid('fc_0_0' )
        f_1_0 = self._format_grid('fc_1_0' )
        f_m1_0= self._format_grid('fc_m1_0')
        f_0_1 = self._format_grid('fc_0_1' )
        f_0_m1= self._format_grid('fc_0_m1')
        self.hm = np.zeros((15,15))
        for ix in range(15):
            for iy in range(15):
                ix0 = ix//3
                ix1 = min(14,ix+1)//3
                ixm = max(0, ix-1)//3
                iy0 = iy//3
                iy1 = min(14,iy+1)//3
                iym = max(0, iy-1)//3
                # print(ix0, ix1, ixm, iy0, iy1, iym)
                self.hm[ix, iy] = 0.2 * (f_0_0 [ix0, iy0] +
                                         f_1_0 [ix1, iy0] +
                                         f_m1_0[ixm, iy0] +
                                         f_0_1 [ix0, iy1] +
                                         f_0_m1[ix0, iym])
        self.full_hm = cv2.resize(self.hm,
                                  self.source.getImage().shape[1::-1],
                                  interpolation=cv2.INTER_CUBIC)


    def _compute_gaze(self):
        coord_max = np.unravel_index(np.argmax(self.full_hm), self.full_hm.shape)
        self.gaze = (coord_max[1]/float(self.source.getImage().shape[1]), coord_max[0]/float(self.source.getImage().shape[0]))
