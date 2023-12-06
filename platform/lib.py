# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This demo lets you to explore the Udacity self-driving car image dataset.
# More info: https://github.com/streamlit/demo-self-driving

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib, cv2

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

    # Download external dependencies.
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("streamlit_app.py"))
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app()

# This file downloader demonstrates Streamlit animation.
def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.
    @st.experimental_memo
    def load_metadata(url):
        return pd.read_csv(url)

    # This function uses some Pandas magic to summarize the metadata Dataframe.
    @st.experimental_memo
    def create_summary(metadata):
        one_hot_encoded = pd.get_dummies(metadata[["frame", "label"]], columns=["label"])
        summary = one_hot_encoded.groupby(["frame"]).sum().rename(columns={
            "label_biker": "biker",
            "label_car": "car",
            "label_pedestrian": "pedestrian",
            "label_trafficLight": "traffic light",
            "label_truck": "truck"
        })
        return summary

    # An amazing property of st.cached functions is that you can pipe them into
    # one another to form a computation DAG (directed acyclic graph). Streamlit
    # recomputes only whatever subset is required to get the right answer!
    metadata = load_metadata(os.path.join(DATA_URL_ROOT, "labels.csv.gz"))
    summary = create_summary(metadata)

    # Uncomment these lines to peek at these DataFrames.
    # st.write('## Metadata', metadata[:1000], '## Summary', summary[:1000])

    # Draw the UI elements to search for objects (pedestrians, cars, etc.)
    selected_frame_index, selected_frame = frame_selector_ui(summary)
    if selected_frame_index == None:
        st.error("No frames fit the criteria. Please select different label or number.")
        return

    # Draw the UI element to select parameters for the YOLO object detector.
    confidence_threshold, overlap_threshold = object_detector_ui()

    # Load the image from S3.
    image_url = os.path.join(DATA_URL_ROOT, selected_frame)
    image = load_image(image_url)

    # Add boxes for objects on the image. These are the boxes for the ground image.
    boxes = metadata[metadata.frame == selected_frame].drop(columns=["frame"])
    draw_image_with_boxes(image, boxes, "Ground Truth",
        "**Human-annotated data** (frame `%i`)" % selected_frame_index)

    # Get the boxes for the objects detected by YOLO by running the YOLO model.
    yolo_boxes = yolo_v3(image, confidence_threshold, overlap_threshold)
    draw_image_with_boxes(image, yolo_boxes, "Real-time Computer Vision",
        "**YOLO v3 Model** (overlap `%3.1f`) (confidence `%3.1f`)" % (overlap_threshold, confidence_threshold))

# This sidebar UI is a little search engine to find certain object types.
def frame_selector_ui(summary):
    st.sidebar.markdown("# Frame")

    # The user can pick which type of object to search for.
    object_type = st.sidebar.selectbox("Search for which objects?", summary.columns, 2)

    # The user can select a range for how many of the selected objecgt should be present.
    min_elts, max_elts = st.sidebar.slider("How many %ss (select a range)?" % object_type, 0, 25, [10, 20])
    selected_frames = get_selected_frames(summary, object_type, min_elts, max_elts)
    if len(selected_frames) < 1:
        return None, None

    # Choose a frame out of the selected frames.
    selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, len(selected_frames) - 1, 0)

    # Draw an altair chart in the sidebar with information on the frame.
    objects_per_frame = summary.loc[selected_frames, object_type].reset_index(drop=True).reset_index()
    chart = alt.Chart(objects_per_frame, height=120).mark_area().encode(
        alt.X("index:Q", scale=alt.Scale(nice=False)),
        alt.Y("%s:Q" % object_type))
    selected_frame_df = pd.DataFrame({"selected_frame": [selected_frame_index]})
    vline = alt.Chart(selected_frame_df).mark_rule(color="red").encode(x = "selected_frame")
    st.sidebar.altair_chart(alt.layer(chart, vline))

    selected_frame = selected_frames[selected_frame_index]
    return selected_frame_index, selected_frame

# Select frames based on the selection in the sidebar
@st.cache(hash_funcs={np.ufunc: str})
def get_selected_frames(summary, label, min_elts, max_elts):
    return summary[np.logical_and(summary[label] >= min_elts, summary[label] <= max_elts)].index

# This sidebar UI lets the user select parameters for the YOLO object detector.
def object_detector_ui():
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold

# Draws an image with boxes overlayed to indicate the presence of cars, pedestrians etc.
def draw_image_with_boxes(image, boxes, header, description):
    # Superpose the semi-transparent object detection boxes.    # Colors for the boxes
    LABEL_COLORS = {
        "car": [255, 0, 0],
        "pedestrian": [0, 255, 0],
        "truck": [0, 0, 255],
        "trafficLight": [255, 255, 0],
        "biker": [255, 0, 255],
    }
    image_with_boxes = image.astype(np.float64)
    for _, (xmin, ymin, xmax, ymax, label) in boxes.iterrows():
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += LABEL_COLORS[label]
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2

    # Draw the header and image.
    st.subheader(header)
    st.markdown(description)
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)

# Download a single file and make its content available as a string.
@st.experimental_singleton(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

# This function loads an image from Streamlit public repo on S3. We use st.cache on this
# function as well, so we can reuse the images across runs.
@st.experimental_memo(show_spinner=False)
def load_image(url):
    with urllib.request.urlopen(url) as response:
        image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]] # BGR -> RGB
    return image

# Run the YOLO model to detect objects.
def yolo_v3(image, confidence_threshold, overlap_threshold):
    # Load the network. Because this is cached it will only happen once.
    @st.cache(allow_output_mutation=True)
    def load_network(config_path, weights_path):
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        output_layer_names = net.getLayerNames()
        output_layer_names = [output_layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layer_names
    net, output_layer_names = load_network("yolov3.cfg", "yolov3.weights")

    # Run the YOLO neural net.
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layer_names)

    # Supress detections in case of too low confidence or too much overlap.
    boxes, confidences, class_IDs = [], [], []
    H, W = image.shape[:2]
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box.astype("int")
                x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_IDs.append(classID)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, overlap_threshold)

    # Map from YOLO labels to Udacity labels.
    UDACITY_LABELS = {
        0: 'pedestrian',
        1: 'biker',
        2: 'car',
        3: 'biker',
        5: 'truck',
        7: 'truck',
        9: 'trafficLight'
    }
    xmin, xmax, ymin, ymax, labels = [], [], [], [], []
    if len(indices) > 0:
        # loop over the indexes we are keeping
        for i in indices.flatten():
            label = UDACITY_LABELS.get(class_IDs[i], None)
            if label is None:
                continue

            # extract the bounding box coordinates
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

            xmin.append(x)
            ymin.append(y)
            xmax.append(x+w)
            ymax.append(y+h)
            labels.append(label)

    boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels})
    return boxes[["xmin", "ymin", "xmax", "ymax", "labels"]]

# Path to the Streamlit public S3 bucket
DATA_URL_ROOT = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"

# External files to download.
EXTERNAL_DEPENDENCIES = {
    "yolov3.weights": {
        "url": "https://pjreddie.com/media/files/yolov3.weights",
        "size": 248007048
    },
    "yolov3.cfg": {
        "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "size": 8342
    }
}

st.download_button(label='Download Image',
                        data= open('yourimage.png', 'rb').read(),
                        file_name='imagename.png',
                        mime='image/png')

if __name__ == "__main__":
    main()


    from argparse import Namespace
import torch
import torch.nn as nn
import numpy as np
import scipy
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from fpgnn.data import GetPubChemFPs, create_graph, get_atom_features_dim,prepare_feature
import csv
import os
import sys
import time
import math
import pandas as pd
import networkx as nx
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import matplotlib.pyplot as plt

#from dgl import functional as fn

atts_out = []


class SelfAttention(nn.Module):
    
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):   
        """
        假设 hidden_size = 128, num_attention_heads = 8, dropout_prob = 0.2
        即隐层维度为128，注意力头设置为8个
        """
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:   # 整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        # 参数定义
        self.num_attention_heads = num_attention_heads    # 8
        self.attention_head_size = int(hidden_size / num_attention_heads)  # 16  每个注意力头的维度
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)   
        # all_head_size = 128 即等于hidden_size, 一般自注意力输入输出前后维度不变
        
        # query, key, value 的线性变换（上述公式2）
        self.query = nn.Linear(hidden_size, self.all_head_size,bias=False)    # 128, 128
        self.key = nn.Linear(hidden_size, self.all_head_size,bias = False)
        self.value = nn.Linear(hidden_size, self.all_head_size,bias = False)
        
        # dropout
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  假设hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)   # 
        return x.permute(0, 2, 1, 3)   # [bs, 8, seqlen, 16]

    def forward(self, hidden_states, attention_mask=None):

        # 线性变换
        mixed_query_layer = self.query(hidden_states)   # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(hidden_states)       # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(hidden_states)   # [bs, seqlen, hid_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)    # [bs, 8, seqlen, 16]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)   # [bs, 8, seqlen, 16]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # 计算query与title之间的点积注意力分数，还不是权重（个人认为权重应该是和为1的概率分布）
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)   # [bs, 8, seqlen, seqlen]
        # 除以根号注意力头的数量，可看原论文公式，防止分数过大，过大会导致softmax之后非0即1

        # 将注意力转化为概率分布，即注意力权重
        attention_probs = nn.Softmax(dim=-1)(attention_scores)    # [bs, 8, seqlen, seqlen]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        # 矩阵相乘，[bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 16] = [bs, 8, seqlen, 16]
        context_layer = torch.matmul(attention_probs, value_layer)   # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()   # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)   # [bs, seqlen, 128]
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer#+hidden_states    # [bs, seqlen, 128] 得到输出

class FPAN(nn.Module):

    def __init__(self,args):
        super(FPAN,self).__init__()
        self.args = args
        
        self.input_dim=512

        self.maccslen=167
        self.pubchemlen=881
        self.ecfplen=1024
        self.erglen=441
        self.maxsize=896
        self.num_heads=8

        self.attention=SelfAttention(self.maxsize,8,args.sadp)
        self.attention2=SelfAttention(self.maxsize,8,args.sadp)

        self.fp_2_dim=args.fp_2_dim
        self.dropout_fpn = args.dropout
        self.cuda = args.cuda
        self.hidden_dim = args.hidden_size

        self.smile2fp= {}

        self.fc1=nn.Linear(3*self.maxsize, self.hidden_dim)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim)
        self.dropout = nn.Dropout(p=self.dropout_fpn)

    def forward(self, smile):
        fpbatch=torch.empty(size=(len(smile),3,self.maxsize))
        fpbatch = fpbatch.cuda()
        for i, one in enumerate(smile):

            if one in self.smile2fp:
                fps=self.smile2fp[one]
            else:
                fps=[]
                mol = Chem.MolFromSmiles(one)
                fp_maccs = list(AllChem.GetMACCSKeysFingerprint(mol))+[0]*(self.maxsize-self.maccslen)
                fp_erg = list(AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1))+[0]*(self.maxsize-self.erglen)
                fp_pubchem = list(GetPubChemFPs(mol))+[0]*(self.maxsize-self.pubchemlen)
                #fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fps.extend(fp_maccs+fp_erg+fp_pubchem)

                self.smile2fp[one]=fps
            fp_tensor=torch.Tensor(fps).view(3,-1)

        
            if self.cuda:
                fp_tensor=fp_tensor.cuda()
 
            fpbatch[i]=fp_tensor

        #print(query.device)
        fpn_out= self.attention(fpbatch)
        #fpn_out= self.attention2(fpn_out)
        
        fpn_out = fpn_out.view(-1,3*self.maxsize)

        fpn_out = self.fc1(fpn_out)
        return fpn_out



class GATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        mode = "add",
        att = "edge",
        feat_drop=0.,
        attn_drop=0.,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super(GATConv, self).__init__()
        self.num_heads = num_heads
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.edge_feats = 7
        self._allow_zero_in_degree = allow_zero_in_degree
        self.elength = out_feats
        self.mode = mode
        if mode == "cross":
            self.embeding = 3
        else:
            self.embeding = 32
        
        self.att = att
        #self.belta = nn.Parameter(torch.FloatTensor(1),requires_grad = True)


        #self.fc = nn.Linear( self._in_src_feats, out_feats * num_heads, bias=False)

        self.q = nn.Linear( self.in_feats, self.embeding * num_heads, bias=False)
        self.k = nn.Linear( self.in_feats, self.embeding * num_heads, bias=False)
        self.v = nn.Linear( self.in_feats, out_feats * num_heads,bias=False)

        if att == "total":
            self.bond_left_fc = nn.Linear(
                self.num_heads*self.embeding,
                self.num_heads*self.elength,
                bias = True)

            self.bond_right_fc = nn.Linear(
                self.num_heads*self.embeding,
                self.num_heads*self.elength,
                bias = True)
        
        self.edge_fc = nn.Linear(self.edge_feats,self.num_heads*self.out_feats,bias=True)

        self.edge_k = nn.Linear(self.edge_feats,self.num_heads*self.embeding,bias=True)
        #self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        #self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = nn.Dropout(0.2)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.has_linear_res = False
        self.has_explicit_bias = False
        if residual:
            if self.in_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(self.in_feats, num_heads * out_feats, bias=bias)
                self.has_linear_res = True
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation
        self.act = nn.functional.elu
        #self.activation = torch.nn.functional.elu
        

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.v.weight, gain=gain)
        nn.init.xavier_normal_(self.q.weight, gain=gain)
        nn.init.xavier_normal_(self.k.weight, gain=gain)
        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value
    
    def edge_function(self,edges):
        crossproduct = torch.cross(edges.src['query'],edges.dst['key'])
        epoint = (crossproduct*edges.data['edge_key']).sum(dim=-1).unsqueeze(-1)
        return {'e':epoint}

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            n = feat.shape[0]

            h_src = h_dst = self.feat_drop(feat)

            feat_src  = self.v(h_src).view(
                n, self.num_heads, self.out_feats
                )
            #el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            #er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            edge_feat = graph.edata['bond']
            edge_key = self.edge_k(edge_feat).view(-1,self.num_heads,self.embeding)
            graph.edata.update({"edge_key":edge_key})

            src_embeding = self.q(h_src)
            dst_embeding = self.k(h_src)

            query = src_embeding.view(n,self.num_heads,self.embeding)
            key = dst_embeding.view(n,self.num_heads,self.embeding)
            #e=(query*key).sum(dim=-1).unsqueeze(-1)
            graph.ndata.update({"ft":feat_src,"query":query,"key":key})

            #graph.ndata.update({"ft": feat_src, "ek": el})
            #graph.ndata.update({"er": er})]
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            #*****graph.apply_edges(fn.u_mul_v("query", "key", "e"))
            #graph.apply_edges(fn.u_dot_v("query", "key", "e"))
            #***graph.edata['e'] = (graph.edata['e']*graph.edata['edge_key']).sum(dim=-1).unsqueeze(-1)
            #e = self.leaky_relu(graph.edata.pop("e").sum(dim=-1).unsqueeze(-1))

            #e = self.leaky_relu(graph.edata.pop("e").unsqueeze(-1))
            
            """
            mul   (a*b)*c
            add   (a+b)*c
            cross   (axb)*c
            """
            if self.mode =="cross":
                graph.apply_edges(self.edge_function)
            elif self.mode == "mul":
                graph.apply_edges(fn.u_mul_v("query", "key", "e"))
                graph.edata['e'] = (graph.edata['e']*graph.edata['edge_key']).sum(dim=-1).unsqueeze(-1)
            else:
                graph.apply_edges(fn.u_add_v("query","key","e"))
                graph.edata['e'] = (graph.edata['e']*graph.edata['edge_key']).sum(dim=-1).unsqueeze(-1)

            # compute softmax
            e = graph.edata.pop("e")
            #e = self.leaky_relu(e)
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            #print("a size",graph.edata['a'].size())
            #print("a tensor",graph.edata['a'])
            # message passing
            
            bond_feat = graph.edata.pop("bond")
            edge_feat = self.edge_fc(bond_feat).view(-1,self.num_heads,self.out_feats)
            graph.edata.update({"edge_feat":edge_feat})

            if self.att == "total":
                src_embeding = self.act(src_embeding)
                dst_embeding = self.act(dst_embeding)
                bond_src = self.bond_left_fc(src_embeding).view(-1,self.num_heads,self.elength)
                bond_dst = self.bond_right_fc(dst_embeding).view(-1,self.num_heads,self.elength)
                graph.ndata.update({"bond_src":bond_src,"bond_dst":bond_dst})

                #graph.apply_edges(fn.u_add_v("bond_src", "bond_dst", "x"))
                graph.ndata["zeros"] = torch.zeros_like(graph.ndata["bond_src"])

                #graph.apply_edges(fn.u_add_v("zeros","bond_src","x"))
                #graph.apply_edges(fn.e_add_v("x","bond_src","x"))
                #graph.apply_edges(fn.u_add_v( "edge_feat","bond_key", "edge_feat"))
                graph.apply_edges(fn.copy_src("bond_dst","x"))
                graph.edata["edge_feat"] = graph.edata["edge_feat"]*graph.edata["x"]

            #graph.apply_edges(fn.u_add_e("ft", "edge_feat", "message"))
            #
            graph.apply_edges(fn.u_mul_e("ft", "edge_feat", "message"))
            graph.edata['message'] = graph.edata['message']*graph.edata['a']
            graph.update_all(fn.copy_e("message","m"), fn.sum("m", "ft"))

            #graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.ndata["ft"]
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(
                    n, -1, self.out_feats
                )
                rst = rst + resval
            # bias
            if self.has_explicit_bias:
                rst = rst + self.bias.view(
                    -1,
                    self.num_heads,
                    self.out_feats
                )
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst

class MPNN(nn.Module):
    def __init__(self,in_features,out_features):
        super(MPNN,self).__init__()
        self.fc1=nn.Linear(in_features,out_features,bias = True)
        self.fc2=nn.Linear(out_features,out_features)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.attfc = nn.Linear(in_features,1,bias = False)

        self.belta = nn.Parameter(torch.FloatTensor(1),requires_grad = True)
        #self.fc3 = nn.Linear(out_features,out_features)
        
    
    def forward(self,graph,feat):
        with graph.local_scope():
            feat_src=self.fc1(feat)

            feat_src=self.leaky_relu(feat_src)
            weight = self.belta/graph.edata.pop("dist")

            
            graph.ndata.update({"feat_src":feat_src})
            #print(graph.ndata['feat_src'])
            graph.edata.update({"weight":weight})

            graph.update_all(fn.u_mul_e("feat_src", "weight", "m"), fn.sum("m", "ft"))
            e = self.attfc(feat)
            graph.ndata['e'] = e

            #rst = graph.ndata["ft"]+(1+self.e)*feat
            rst = graph.ndata["ft"]+graph.ndata['e']*feat
            rst = self.fc2(rst)
            rst = self.leaky_relu(rst)
            #rst = nn.functional.relu(rst)
            return rst

class GNN(nn.Module):
    def __init__(self,args):
        super(GNN,self).__init__()
        self.cuda = args.cuda
        self.args = args
        self.nheads = 8
        self.smile2graph={}
        self.smile2dist={}
        self.smile2edge = {}
        self.atom_feat_num = 32
        self.fc = nn.Linear(self.atom_feat_num*self.nheads*6,args.hidden_size,bias = True)
        self.sum=0

        self.gat = GATConv(89,self.atom_feat_num,self.nheads,
                    feat_drop = 0.0,
                    attn_drop = 0.,
                    mode = args.mode,
                    att = args.att,
                    #activation= nn.functional.elu,
                   )
        self.gat2 = GATConv(self.atom_feat_num*self.nheads,
                    self.atom_feat_num*self.nheads,1,
                    mode = args.mode,
                    att = args.att,
                    feat_drop=0.,attn_drop=0.,
                    residual = True,
                    activation= nn.functional.elu,
                    )
        self.mpnn = MPNN(self.atom_feat_num*self.nheads,
                   self.atom_feat_num*self.nheads
                )
        self.prepare_data()
    
    def prepare_data(self):
        data_path = self.args.data_path
        dataset = path.split("/")[-1][:-4]

        
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists(os.path.join("data",'{}.graph'.format(dataset))):
            self.generate_graph(data_path)

        with open(os.path.join("data",'{}.graph'.format(dataset)),"rb") as f:
            self.smile2graph = pickle.load(f)
        
        with open(os.path.join("data",'{}.distgraph'.format(dataset)),"rb") as f:
            self.smile2dist = pickle.load(f)               


    def get_bond_feat(self,mol,graph,distmatrix):
        e = graph.number_of_edges()

        src,dst = graph.edges()
        edge_feats = []
        bond_type = {"SINGLE":1,"DOUBLE":2,"TRIPLE":3,"AROMATIC":4}
        for index in range(e):
            a = int(src[index])
            b = int(dst[index])
            bond_feat = [0]*7
            if a == b :
                bond_feat[0]=1
            else:
                bond = mol.GetBondBetweenAtoms(a,b)
                btype = str(bond.GetBondType())
                idx =  1 if btype not in bond_type else bond_type[btype]
                bond_feat[idx] = 1
                if bond.IsInRing():
                    bond_feat[5]=1
                dist = distmatrix[a][b]
                bond_feat[6] = dist
                #print(dist)
            edge_feats.append(bond_feat)

        return edge_feats

    def generate_graph(self,data_path):
        df = pd.read_csv(data_path)
        smiles = df.iloc[1:,0]
        dataset  = path.split("/")[-1][:-4]
        path = os.path.join("data","{}.feature".format(dataset))
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)
            mol = Chem.AddHs(mol)
            adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
            adj[np.diag_indices_from(adj)] = 1
            graph = scipy.sparse.coo_matrix(adj)
            #self.smile2graph[smile] = graph

            if AllChem.EmbedMolecule(mol) == -1:
                Chem.rdmolops.RemoveStereochemistry(mol)
                if AllChem.EmbedMolecule(mol,useRandomCoords=True,maxAttempts=10) == -1 :
                    dist = Chem.GetDistanceMatrix(mol)*1.25
                    print("emebed failed,*****************************")
                else:
                    dist=AllChem.Get3DDistanceMatrix(mol)
            else:
                dist=AllChem.Get3DDistanceMatrix(mol)

            bond_feat = get_bond_feat(mol,graph,dist)
            edge_feat_tensor = torch.Tensor(bond_feat)
            graph.edata['bond'] = edge_feat_tensor
            self.smile2graph[smile] = graph

            hop2adj=np.linalg.matrix_power(adj,2)
            hop2adj[hop2adj>0]=1

            hopk=1-hop2
            dist[dist>5.]=0
            dist = hopk*dist*dist
            distweight = scipy.sparse.coo_matrix(dist)
            distgraph = dgl.from_scipy(distweight,eweight_name="dist")
            self.smile2dist[smile]=distgraph

            with open("data","{}.graph".format(dataset),"wb") as f:
                pickle.dump(self.smile2graph,f)
            
            with open("data","{}.distgraph".format(dataset),"wb") as f:
                pickle.dump(self.smile2dist,f)



    def forward(self,smiles):
        #smiles = ["Cc1cc(cc(c1)O)C"]
        mols = create_graph(smiles, self.args)
        atom_feature, atom_index = mols.get_feature()
        if self.cuda:
            atom_feature = atom_feature.cuda()
        graphs = []
        distgraphs=[]
        for smile in smiles:
            graph = self.smile2graph[smile]
            distgraph = self.smile2dist[smile]            
            graphs.append(g)
            distgraphs.append(distg)
       
        biggraph = dgl.batch(graphs).to("cuda")
        distgraph = dgl.batch(distgraphs).to("cuda")
        distgraph = dgl.to_float(distgraph)

        batch_feature = self.gat(biggraph,atom_feature[1:]).view(-1,self.atom_feat_num*self.nheads)
        biggraph.ndata['gat_feat']=self.gat2(biggraph,batch_feature).view(-1,self.atom_feat_num*self.nheads)
        #biggraph.ndata['gat_feat'] = batch_feature

        #gat_outs = torch.stack(gat_outs, dim=0)
        distgraph.ndata['mpnn_feat'] = self.mpnn(distgraph,biggraph.ndata['gat_feat'])
        gnn_out =torch.cat([dgl.sum_nodes(biggraph, 'gat_feat'),
                           dgl.max_nodes(biggraph, 'gat_feat'),
                           dgl.sum_nodes(distgraph, 'mpnn_feat'),
                           dgl.max_nodes(distgraph, 'mpnn_feat'),
                           dgl.mean_nodes(distgraph, 'mpnn_feat'),
                           dgl.mean_nodes(biggraph, 'gat_feat')
                           ],dim=-1)       

        #print(gnn_out.size())
        
         #if self.args.mean:
           # gnn_out = torch.cat(
                #[gnn_out,
                #dgl.mean_nodes(distgraph, 'mpnn_feat'),
                #dgl.mean_nodes(biggraph, 'gat_feat')],
                #dim = 1
            #)
        #print(self.att)
        #gnn_out =self.att(gnn_out)
        #gnn_out = gnn_out.view(-1,self.atom_feat_num*self.nheads*(4+2*self.args.mean))
        
        return self.fc(gnn_out)


class FpgnnModel(nn.Module):
    def __init__(self,is_classif,gat_scale,cuda,dropout_fpn):
        super(FpgnnModel, self).__init__()
        self.gat_scale = gat_scale
        self.is_classif = is_classif
        self.cuda = cuda
        self.dropout_fpn = dropout_fpn
        if self.is_classif:
            self.sigmoid = nn.Sigmoid()

    def create_gat(self,args):
        self.gnn = GNN(args)
    
    def create_fpn(self,args):

        self.encoder2 = FPAN(args)

    
    def create_scale(self,args):
        #print("createscale")
        linear_dim = int(args.hidden_size)
        #print("linear_dim:",linear_dim)
        
        if self.gat_scale == 0:
            self.fc_fpn = nn.Linear(linear_dim,linear_dim)

    def create_ffn(self,args):
        linear_dim = args.hidden_size
        print(self.gat_scale)
        if self.gat_scale == 1:
            self.ffn = nn.Sequential(
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
                                     )
        elif self.gat_scale == 0:
            self.ffn = nn.Sequential(
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
                                     )

        else:
            self.ffn = nn.Sequential(
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim*2, out_features=linear_dim, bias=False),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=False)
                                     )
    
    def forward(self,input):
        if self.gat_scale == 1:
            output = self.gnn(input)
        elif self.gat_scale == 0:
            output = self.encoder2(input)
        else:
            gat_out = self.encoder3(input)
            fpn_out = self.encoder2(input)
            #print("gatoutsize",gat_out.size())
            #gat_out = self.fc_gat(gat_out)
            gat_out = self.act_func(gat_out)
            #print("gatfuturesize",gat_out.size())
            fpn_out = self.fc_fpn(fpn_out)
            fpn_out = self.act_func(fpn_out)
            
            output = torch.cat([gat_out,fpn_out],axis=1)
            #print(output.size())
        output = self.ffn(output)
        
        if self.is_classif and not self.training:
            output = self.sigmoid(output)
        
        return output

def get_atts_out():
    return atts_out

def FPGNN(args):
    if args.dataset_type == 'classification':
        is_classif = 1
    else:
        is_classif = 0
    model = FpgnnModel(is_classif,args.gat_scale,args.cuda,args.dropout)
    #print("gatscale",args.gat_scale)
    if args.gat_scale == 1:
        model.create_scale(args)
        model.create_gat(args)
        model.create_ffn(args)
    elif args.gat_scale == 0:
        model.create_fpn(args)
        model.create_ffn(args)
    else:
        model.create_gat(args)
        model.create_fpn(args)
        model.create_scale(args)
        model.create_ffn(args)
    
    for param in model.parameters():
        if param.dim() == 1:
            #torch.nn.init.normal_(param)
            nn.init.constant_(param, 0)
        else:
            #torch.nn.init.xavier_uniform_(param, gain=1.0)
            nn.init.xavier_normal_(param)
        
    
    return model

