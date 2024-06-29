import cv2
import numpy as np
import argparse
from numpy_tps2flow import transformer
from numpy_grid_sample import grid_sample
from numpy_upsample import upsample

###跟recrecnet的一样
def get_rigid_mesh(height, width, grid_h=6, grid_w=8): 
    ww = np.matmul(np.ones([grid_h+1, 1]), np.expand_dims(np.linspace(0., float(width), grid_w+1), 0))
    hh = np.matmul(np.expand_dims(np.linspace(0.0, float(height), grid_h+1), 1), np.ones([1, grid_w+1]))
    
    ori_pt = np.concatenate((np.expand_dims(ww, 2), np.expand_dims(hh,2)), axis=2)
    return ori_pt[np.newaxis, :]  ###batchsize=1
###跟recrecnet的一样
def get_norm_mesh(mesh, height, width):
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = np.stack([mesh_w, mesh_h], axis=3) 
    
    return norm_mesh.reshape((1, -1, 2)) 


def warp_with_flow(img, flow):
    #initilize grid_coord
    batch, C, H, W = img.shape
    # coords0 = torch.meshgrid(torch.arange(H), torch.arange(W))
    coords0 = np.meshgrid(np.arange(W), np.arange(H))
    coords0 = np.stack(coords0, axis=0).astype(np.float32)
    coords0 = np.tile(coords0[None], (batch, 1, 1, 1)) ### bs, 2, h, w

    # target coordinates
    target_coord = coords0 + flow

    # normalization
    target_coord_w = target_coord[:,0,:,:]*2./float(W) - 1.
    target_coord_h = target_coord[:,1,:,:]*2./float(H) - 1.
    target_coord_wh = np.stack([target_coord_w, target_coord_h], axis=1)
    #
    warped_img = grid_sample(img, np.transpose(target_coord_wh, (0,2,3,1)), align_corners=True)

    return warped_img

class CoupledTPS_RectanglingNet():
    def __init__(self, modelpatha, modelpathb):
        self.feature_extractor = cv2.dnn.readNet(modelpatha)
        self.input_height, self.input_width = 384, 512
        self.rigid_mesh = get_rigid_mesh(self.input_height, self.input_width)
        self.norm_rigid_mesh = get_norm_mesh(self.rigid_mesh, self.input_height, self.input_width)

        self.regressNet = cv2.dnn.readNet(modelpathb)
        self.input_shape = (1, 256, 24, 32)

    def detect(self, srcimg, mask_img, iter_num):
        img = cv2.resize(srcimg, (self.input_width, self.input_height))
        img = img.astype(np.float32) / 127.5 - 1.0
        input_tensor = cv2.dnn.blobFromImage(img)
        mask = cv2.resize(mask_img, (self.input_width, self.input_height))
        mask = mask.astype(dtype=np.float32) / 255.0
        mask_tensor = cv2.dnn.blobFromImage(mask)

        self.feature_extractor.setInput(input_tensor, "inputa")
        self.feature_extractor.setInput(mask_tensor, "inputb")
        feature_ori = self.feature_extractor.forward(self.feature_extractor.getUnconnectedOutLayersNames())[0]
        feature = feature_ori.copy()
        
        flow = 0
        flow_list = []
        for i in range(iter_num):
            self.regressNet.setInput(feature)
            mesh_motion = self.regressNet.forward(self.regressNet.getUnconnectedOutLayersNames())[0]

            pre_mesh = self.rigid_mesh + mesh_motion
            norm_pre_mesh = get_norm_mesh(pre_mesh, self.input_height, self.input_width)

            delta_flow = transformer(input_tensor, self.norm_rigid_mesh, norm_pre_mesh, (self.input_height, self.input_width))

            if i == 0:
                flow = delta_flow + flow
            else:
                # warp the flow using delta_flow
                warped_flow = warp_with_flow(flow, delta_flow)
                flow = delta_flow + warped_flow
            # save flow
            flow_list.append(flow)

            if i < iter_num-1:
                _, _, fea_h, fea_w = feature.shape

                # downsample the optical flow
                down_flow = upsample(flow, scales=(fea_h/flow.shape[2], fea_w/flow.shape[3]), method="bilinear", align_corners=True)
                down_flow_w = down_flow[:,0,:,:]*fea_w / self.input_width
                down_flow_h = down_flow[:,1,:,:]*fea_h / self.input_height
                down_flow = np.stack([down_flow_w, down_flow_h], axis=1)

                # warp features
                feature = warp_with_flow(feature_ori, down_flow)

        correction_list = []
        #pre_mesh_list = []

        for i in range(iter_num):
            # warp tilted image
            warped_img = warp_with_flow(input_tensor, flow_list[i])
            # list appending
            correction_list.append(warped_img)


        out_dict = {}
        out_dict.update(correction = correction_list, flow_list = flow_list)

        correction_final = out_dict['correction'][-1]

        correction_np = np.transpose((correction_final[0]+1)*127.5, (1,2,0)).astype(np.uint8)
        return correction_np
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='testimgs/rectangling/input/00474.jpg')
    parser.add_argument("--maskpath", type=str, default='testimgs/rectangling/mask/00474.jpg')
    parser.add_argument('--iter_num', type=int, default=3)
    args = parser.parse_args()

    myNet = CoupledTPS_RectanglingNet('weights/rectangling/feature_extractor.onnx', 'weights/rectangling/regressnet.onnx')
    srcimg = cv2.imread(args.imgpath)
    ori_height, ori_width, _ = srcimg.shape 
    mask_img = cv2.imread(args.maskpath)
    pred_out = myNet.detect(srcimg, mask_img, args.iter_num)

    if ori_height>=ori_width:
        mid_img = np.zeros((ori_height, 20, 3), dtype=np.uint8)+255
        combine_img = np.hstack((srcimg, mid_img, pred_out))
    else:
        mid_img = np.zeros((20, ori_width, 3), dtype=np.uint8)+255
        combine_img = np.vstack((srcimg, mid_img, pred_out))
    
    ####cv2.imwrite('python-opencv-dnn-CoupledTPS_Rectangling.jpg', combine_img)
    cv2.namedWindow('opencv-dnn-CoupledTPS_Rectangling', 0)
    cv2.imshow('opencv-dnn-CoupledTPS_Rectangling', combine_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()