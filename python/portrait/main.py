import cv2
import numpy as np
import argparse
from numpy_tps2flow import transformer
from numpy_grid_sample import grid_sample
from numpy_upsample import upsample

def get_rigid_mesh(height, width, grid_h=7, grid_w=10): 
    ww = np.matmul(np.ones([grid_h+1, 1]), np.expand_dims(np.linspace(0., float(width), grid_w+1), 0))
    hh = np.matmul(np.expand_dims(np.linspace(0.0, float(height), grid_h+1), 1), np.ones([1, grid_w+1]))
    
    ori_pt = np.concatenate((np.expand_dims(ww, 2), np.expand_dims(hh,2)), axis=2)

    # to discard some points
    grid_index = (ori_pt[:, :, 1] >=(192-75) ) & (ori_pt[:, :, 1] <= (192+75))
    grid_index = grid_index & (ori_pt[:, :, 0] >= (256-100)) & (ori_pt[:, :, 0] <= (256+100))
    grid_index = ~grid_index
    grid = ori_pt[grid_index]

    return grid[np.newaxis, :]  ###batchsize=1

def get_norm_mesh(mesh, height, width):
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = np.stack([mesh_w, mesh_h], axis=2) 
    
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

class CoupledTPS_PortraitNet():
    def __init__(self, modelpatha, modelpathb):
        self.feature_extractor = cv2.dnn.readNet(modelpatha)
        self.input_height, self.input_width = 384, 512
        self.output_names = self.feature_extractor.getUnconnectedOutLayersNames()
        self.rigid_mesh = get_rigid_mesh(self.input_height, self.input_width)
        self.norm_rigid_mesh = get_norm_mesh(self.rigid_mesh, self.input_height, self.input_width)

        self.regressNet = cv2.dnn.readNet(modelpathb)
        self.input_shape = (1, 256, 24, 32)

    def detect(self, srcimg, iter_num):
        ori_height, ori_width, _ = srcimg.shape 
        input = srcimg.copy()
        img = cv2.resize(srcimg, dsize=(self.input_width, self.input_height))
        img = img.astype(np.float32) / 127.5 - 1.0
        input_tensor = cv2.dnn.blobFromImage(img)
        self.feature_extractor.setInput(input_tensor)
        feature_ori = self.feature_extractor.forward(self.output_names)[0]
        feature = feature_ori.copy()
        
        flow = 0
        flow_list = []
        norm_pre_mesh_list = []
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
            norm_pre_mesh_list.append(norm_pre_mesh)


            if i < iter_num-1:
                _, _, fea_h, fea_w = feature.shape

                # downsample the optical flow
                # down_flow = cv2.resize(flow, dsize=(fea_h, fea_w), interpolation='bilinear', align_corners=True) ###opencv的resize函数没有align_corners参数，默认是False的
                down_flow = upsample(flow, scales=(fea_h/flow.shape[2], fea_w/flow.shape[3]), method="bilinear", align_corners=True)
                down_flow_w = down_flow[:,0,:,:]*fea_w / self.input_width
                down_flow_h = down_flow[:,1,:,:]*fea_h / self.input_height
                down_flow = np.stack([down_flow_w, down_flow_h], axis=1)

                # warp features
                feature = warp_with_flow(feature_ori, down_flow)

        correction_list = []

        for i in range(iter_num):
            # warp tilted image
            warped_img = warp_with_flow(input_tensor, flow_list[i])
            # list appending
            correction_list.append(warped_img)

        out_dict = {}
        out_dict.update(correction = correction_list, flow_list = flow_list, norm_pre_mesh_list = norm_pre_mesh_list)
        
        pred = out_dict['flow_list'][-1].squeeze(0)   ###只有flow_list有用, 并且是列表里的最后一个才有用, c++程序可以简化了
        pflow = pred.transpose(1, 2, 0)
        predflow_x, predflow_y = pflow[:, :, 0], pflow[:, :, 1]

        scale_x = ori_width / predflow_x.shape[1]
        scale_y = ori_height / predflow_x.shape[0]
        predflow_x = cv2.resize(predflow_x, (ori_width, ori_height)) * scale_x
        predflow_y = cv2.resize(predflow_y, (ori_width, ori_height)) * scale_y

        # Get the [predicted image]"""
        ys, xs = np.mgrid[:ori_height, :ori_width]
        mesh_x = predflow_x.astype("float32") + xs.astype("float32")
        mesh_y = predflow_y.astype("float32") + ys.astype("float32")
        pred_out = ((cv2.remap((input/255)-1, mesh_x, mesh_y, cv2.INTER_LINEAR)+1)*255).astype(np.uint8)
        return pred_out


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='testimgs/portrait/0083_mi9.jpg')
    parser.add_argument('--iter_num', type=int, default=1)
    args = parser.parse_args()

    myNet = CoupledTPS_PortraitNet('weights/portrait/feature_extractor.onnx', 'weights/portrait/regressnet.onnx')
    srcimg = cv2.imread(args.imgpath)
    ori_height, ori_width, _ = srcimg.shape 
    pred_out = myNet.detect(srcimg, args.iter_num)

    if ori_height>=ori_width:
        mid_img = np.zeros((ori_height, 20, 3), dtype=np.uint8)+255
        combine_img = np.hstack((srcimg, mid_img, pred_out))
    else:
        mid_img = np.zeros((20, ori_width, 3), dtype=np.uint8)+255
        combine_img = np.vstack((srcimg, mid_img, pred_out))
    ####cv2.imwrite('python-opencv-dnn-CoupledTPS_Portrait.jpg', combine_img)
    cv2.namedWindow('opencv-dnn-CoupledTPS_Portrait', 0)
    cv2.imshow('opencv-dnn-CoupledTPS_Portrait', combine_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()