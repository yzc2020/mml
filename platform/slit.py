import streamlit as st

@st.cache_data
def pred(prop,smiles,mode,ntask):
    import deepchem as dc
    import pandas as pd
    import numpy as np
    import os

    featurizer = dc.feat.ConvMolFeaturizer()
    X = featurizer.featurize(smiles)
    dataset = dc.data.NumpyDataset(X=X)
    model_dir = os.path.join("models",prop)
    model = dc.models.GraphConvModel(
      n_tasks = ntask,
      batch_size=4,
      mode=mode,
      model_dir = model_dir
      )

    model.restore()

    output = model.predict(dataset)
    if ntask == 1:
        cols = [prop]
    else:   
        cols = [prop+"_"+str(i) for i in range(ntask)]
    
    
    if mode =="regression":
        result = pd.DataFrame(output,columns = cols)
        #3st.write(result)
    else:
        result = np.argmax(output,axis = -1)
        #cols = [prop+str(i) for i in range(ntask)]
        result = pd.DataFrame(result,columns = cols)
        #st.write(result)
    return result

def test():
    import streamlit as st
    st.balloons()   
    st.markdown("Here is a test page")
    import pandas as pd
    import base64
    from rdkit import Chem
    from rdkit.Chem import Draw
    import os
    import altair as alt
    import numpy as np
    import math
    import time
    tb = st.sidebar.selectbox("Choose the Table",["Datasets","Methods"])
    if tb == "Datasets":
        df = pd.read_csv("./dataset/data.csv",encoding='gb18030')
        st.write(df)
    else:
        df = pd.read_csv("model.csv")
        st.write(df)
        papers = ["GC","MPNN","DMPNN","GAT","GCN","AttentiveFP","MV-ChemNet","GLAM"]
        paper = st.sidebar.selectbox("Method detail",papers)
        if paper:
            path = os.path.join("paper",paper)+".pdf"
            with open(path,"rb") as f:
            #base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="900" height="800" type="application/pdf"></iframe>'
            #pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
            #pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="400" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)


def save_image(file):
    import streamlit as st
    from io import BytesIO
    btn = st.sidebar.download_button(
            label="Download image",
            data =file.tobytes(),
            file_name="flower.png",
            mime="image/png"
        )

def data_visualization():
    import streamlit as st
    import time
    import os
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem  import AllChem
    def setting():
        addN = st.sidebar.radio(
        "Add Atom Index",
        ('Remove', 'Add'))
        addH = st.sidebar.radio(
            "Add H Atom",
            ('Remove','Add')
        )
        return addN,addH
    def plotting(smiles,addN,addH,allow_download=False):
        for smile in smiles:
            mol = Chem.AllChem.MolFromSmiles(smile)
            if addH == 'Add':
                mol = Chem.AddHs(mol)
            if addN == 'Add':
                for atom in mol.GetAtoms():
                    atom.SetProp("atomNote", str(atom.GetIdx()))

            image = Draw.MolToImage(mol,size = [300,300])
            st.image(image,caption=smile)
            if allow_download:
                save_image(image)

    def visualize_data_from_smile():
        #st.write("paste a smiles")
        smile = st.text_input('SMILES')
        
        if not smile:
           st.stop()
           st.warning("please input a SMILES of a melecule")
        addN,addH = setting()
        plotting([smile],addN,addH,True)

    def visualize_data_from_dataset():
        fieds = os.listdir("dataset")
        field_name = st.sidebar.selectbox("Choose the category",fieds)
        datasets = os.listdir(os.path.join("dataset",field_name))
        dataset = st.sidebar.selectbox("Choose the dataset",datasets)
        addN,addH = setting()
        frame = pd.read_csv(os.path.join("dataset",field_name,dataset))
        st.write(frame)
        smiles = frame.iloc[1:11,0]
        plotting(smiles,addN,addH)
    def visualize_data_from_csv():
        uploaded_file = st.sidebar.file_uploader("Choose a scv file")
        if uploaded_file is None:
            st.stop()
        addN,addH = setting()
        dataframe = pd.read_csv(uploaded_file,header = None)
    
        smiles = dataframe.iloc[0:10,0]
        n = len(smiles)
        all_smiles = dataframe.iloc[0:,0]
        bond_sum = 0
        atom_sum = 0
        for smile in all_smiles:
            mol = Chem.MolFromSmiles(smile)

            mol=Chem.AddHs(mol)
            atom_num = mol.GetNumAtoms()

            bond_num = mol.GetNumBonds()
            bond_sum+=bond_num
            atom_sum+=atom_num
        
        st.write("Molecule number:",n)
        st.write("Average atom  number:",int(atom_sum/n))
        st.write("Average bond  number:",int(bond_sum/n))
        st.write(smiles)
        plotting(smiles,addN,addH)
    
    data_type = st.sidebar.selectbox("Choose the Data",["SMILES","Dataset","Upload csv file"])
    if data_type == "SMILES":
        visualize_data_from_smile()
    elif data_type == "Dataset":
        visualize_data_from_dataset()
    else:
        visualize_data_from_csv()


def predict():
    import streamlit as st
    st.balloons()   
    import pandas as pd
    import base64
    from rdkit import Chem
    from rdkit.Chem import Draw
    import os
    import altair as alt
    import numpy as np
    import math
    import time
    import plotly.express as px
    import deepchem as dc
    st.title("Prediction")
    
    properties  = [
        "ESOL",
        "FreeSolv",
        "BBBP",
        "Lipophilicity",
        "HIV",
        "BACE",
        "Clintox",
        "MUV",
        "Tox21",
        "SIDER",
    ]
    reg_task =["ESOL","FreeSolv","Lipophilicity","PDBbind"]
    cla_task = ["BBBP","PCBA","HIV","BACE", "Clintox","MUV","Tox21","SIDER"]
    ntask =1
    

    predict_csv = st.file_uploader("Upload smiles csv")
    st.write()

    selected_prop = [
            prop
            for prop in properties
            if st.sidebar.checkbox(prop, False)
        ]
    p = st.sidebar.checkbox("Predict")
    if predict_csv:
        dataframe = pd.read_csv(predict_csv,header = None)
        #st.write(dataframe)    
    if not p or not predict_csv:
        st.stop()
    
    smiles = dataframe.iloc[:,0]
    df = pd.DataFrame({"smiles":smiles})
    if selected_prop:
        for prop in selected_prop:
            if prop in cla_task:
                mode = "classification"
            if prop in reg_task:
                mode = "regression"
            
            ntask = 1
            
            if prop == "PCBA":
                ntask = 128
            if prop == "Clintox":
                ntask = 2
            if prop == "MUV":
                ntask = 17
            if prop == "Tox21":
                ntask = 12
            if prop == "SIDER":
                ntask == 27

            
            result = pred(prop,smiles,mode,ntask)
            df = pd.concat([df,result],axis = 1)
        
    else:
        st.error("Please choose at least one property")
    
    st.write(df)

    st.download_button(
        label="Download data as CSV",
        data=df.to_csv(index = False).encode('utf-8'),
        file_name='property.csv',
        mime='text/csv',
    )


def performance():
    import streamlit as st
    st.balloons()   
    import pandas as pd
    import base64
    from rdkit import Chem
    from rdkit.Chem import Draw
    import os
    import altair as alt
    import numpy as np
    import math
    import time
    
    tb = st.sidebar.selectbox("Choose the Table",["Datasets","Methods"])
    if tb == "Datasets":
        st.title("Dataset Details")
        df = pd.read_csv("./dataset/data.csv",encoding='gb18030')
        st.write(df)
    else:
        st.title("Performance")
        df = pd.read_csv("model.csv")
        st.write(df) 
        papers = ["GC","MPNN","DMPNN","GAT","GCN","AttentiveFP","ChemNet","GLAM"]
        paper = st.sidebar.selectbox("Method detail",papers)
        if paper:
            path = os.path.join("paper",paper)+".pdf"
            with open(path,"rb") as f:
            #base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="900" height="800" type="application/pdf"></iframe>'
            #pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
            #pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="400" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)


def training():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import os
    import plotly.express as px

    def generate_dataset(dataframe,featurizer):
        smiles = dataframe.iloc[1:,0]
        labels = dataframe.iloc[1:,1]
        #smiles = ["C1CCC1", "C1=CC=CN=C1"]
        #labels = [0., 1.]
        #st.write(smiles)
        #labels = dataframe.iloc[1:,1]*1.0
        X = featurizer.featurize(smiles)
        dataset = dc.data.NumpyDataset(X = X, y = labels)
        return dataset

    data_type = st.sidebar.selectbox("Choose the Data",["Dataset","Upload csv file"])

    if data_type != "Dataset":
        col1, col2 = st.columns(2)
        loss_container = col1.empty()
        perf_container = col2.empty()

        train_csv = col1.file_uploader("Choose a scv train set")
        valid_csv = col2.file_uploader("Choose a scv valid or test set(optional)")

        check = st.checkbox('File Check')

        if not check:
            st.stop()
        train_dataframe = pd.read_csv(train_csv)

        if valid_csv != None:
            valid_dataframe = pd.read_csv(valid_csv)
        else:
            valid_dataframe = None

    else:
        col1, col2 = st.columns(2)
        loss_container = col1.empty()
        perf_container = col2.empty()

        field = col1.selectbox("Choose the Category",["MoleculeNet","BreastCellLines"])
        if field =="MoleculeNet":
            datasetname = col2.selectbox("Choose the dataset",
            ["Freesolv",
            "BACE (Regression)",
             "BACE (Classification)",
             "BBBP",
             "Clintox",
             "Delaney(Esol)",
             "HIV",
             "LIPO",
             "MUV",
             "PCBA",
             "PDBBIND",
             "QM7",
             "QM8",
             "QM9",
             "SIDER",
             "Tox21",
             "Toxcast"])
        else:
            filenames = os.listdir(field)
            datasetname = col2.selectbox("Choose the dataset",filenames)
            train_dataframe = pd.read_csv(os.path.join("BreastCellLines",datasetname))
            st.write(train_dataframe)
            valid_dataframe = None

        if not datasetname:
            st.stop()
        
    task_type = st.sidebar.selectbox("Task type",["regression","classification"])
    task_num = st.sidebar.number_input(" Task num",value =1)
    
    epoch = st.sidebar.number_input("Epoch",value = 300)
    bs = st.sidebar.number_input("Batch size",value = 4)
    #lr = st.sidebar.number_input("Initial Learning Rate",value = 0.001)
    lr = st.sidebar.text_input("Initial learning rate",value = "1e-3")
    lr = float(lr)


    import deepchem as dc

    model2feat = {
        "GCNModel":"MolGraphConvFeaturizer",
        "GATModel":"MolGraphConvFeaturizer",
        "AttentiveFPModel":"MolGraphConvFeaturizer",
        "MPNNModel":"MolGraphConvFeaturizer",
        "GraphConvModel":"ConvMolFeaturizer",
        "DMPNNModel":"DMPNNFeaturizer",
        "WeaveModel":"WeaveFeaturizer",
        "ChemNet":"MolGraphConvFeaturizer",
        "GNNModular":"SNAPFeaturizer",
        "CGCNNModel":"CGCNNFEaturizer",
        "AtomicConvModel":"ComplexNeighborListFragmentAtomicCoordinates",
        "ChemCeption":"SmilesToImage",
        "PagtnModel":"PagtnMolGraphFeaturizer MolGraphConvFeaturizer",
        "Smiles2Vec":"SmilesToSeq",
        "DTNNModel":"CoulombMatrix",
        "BasicMolGANModel":"MolGanFeaturizer",
        "LCNNModel":"LCNNFeaturizer",
    }

    model2model = {
        "GCNModel":dc.models.GCNModel,
        "ScScoreModel":dc.models.ScScoreModel,
        "AtomicConvModel":dc.models.AtomicConvModel,
        "AttentiveFPModel":dc.models.AttentiveFPModel,
        "ChemCeption":dc.models.ChemCeption,
        "GraphConvModel":dc.models.GraphConvModel,
        "MPNNModel":dc.models.MPNNModel,
        "GATModel":dc.models.GATModel,
        "PagtnModel":dc.models.PagtnModel,
        "Smiles2Vec":dc.models.Smiles2Vec,
        "DTNNModel":dc.models.DTNNModel,
        "WeaveModel":dc.models.WeaveModel,
        "BasicMolGANModel":dc.models.BasicMolGANModel,
        "DMPNNModel":dc.models.DMPNNModel,
        "CGCNNModel":dc.models.CGCNNModel,
        "LCNNModel":dc.models.LCNNModel,
    }

    model_choice = st.sidebar.selectbox("Choose the model",model2feat.keys())

    check = st.sidebar.checkbox('Check',value = False)

    if not check:
        st.stop()

    import deepchem as dc

    dataset2load = {
        "BACE (Regression)":dc.molnet.load_bace_classification,
        "BACE (Classification)":dc.molnet.load_bace_regression,
        "BBBP":dc.molnet.load_bbbp,
        "Clintox":dc.molnet.load_clintox,
        "Delaney(Esol)":dc.molnet.load_delaney,
        "Freesolv":dc.molnet.load_freesolv,
        "HIV":dc.molnet.load_hiv,
        "LIPO":dc.molnet.load_lipo,
        "MUV":dc.molnet.load_muv,
        "PCBA":dc.molnet.load_pcba,
        "PDBBIND":dc.molnet.load_pdbbind,
        "QM7":dc.molnet.load_qm7,
        "QM8":dc.molnet.load_qm8,
        "QM9":dc.molnet.load_qm9,
        "SIDER":dc.molnet.load_sider,
        "Tox21":dc.molnet.load_tox21,
        "Toxcast":dc.molnet.load_toxcast,
    }
    if model_choice == "ChemNet":
        model_choice = "GCNModel"
    if model_choice == "AttentiveFPModel" or model_choice == "MPNNModel" :
        para = "use_edges=True"
    else:
        para = ""
    #st.write(para)

    featurizer = eval("dc.feat."+model2feat[model_choice]+"("+para+")")
    print()
    print(featurizer)

    if field == "MoleculeNet":
        
        tasks,datasets,transformers=dataset2load[datasetname](featurizer=featurizer,reload=False)
        train_dataset,valid_dataset,test_dataset=datasets
        ntask = train_dataset.y.shape[1]
    else:
        train_dataset = generate_dataset(train_dataframe,featurizer)
        test_dataset = None
        if valid_dataframe:
            train_dataset = generate_dataset(train_dataframe,featurizer)
            valid_dataset = generate_dataset(valid_dataframe,featurizer)
        else:
            total_len = len(train_dataframe)-1
            train_dataset = generate_dataset(train_dataframe[0:int(total_len*0.9)],featurizer)
            valid_dataset = generate_dataset(train_dataframe[int(total_len*0.9):],featurizer)

    #model = eval("dc.models."+model_choice)
    modelclass = model2model[model_choice]
    print(modelclass)

    if task_type == "classification":
        metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    else:
        metric = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
    

    model = modelclass(mode = task_type, n_tasks = task_num,
                 batch_size = bs, learning_rate = lr,model_dir = "./tmp")
    
    if task_type == "classification":
        metric_key = "roc_auc_score"
    else:
        metric_key = "mean-rms_score"

    loss = model.fit(train_dataset)
    perf = model.evaluate(valid_dataset,[metric])[metric_key]

    #df = pd.DataFrame(
    #    {"epoch": [1], "loss": [loss], "perf": [perf]}
    #).set_index("epoch")

    df = pd.DataFrame({"epoch": [], "loss": [], "performance": []}).set_index("epoch")

    #loss_fig = px.line(df, x =df.index, y=["loss"])
    #perf_fig = px.line(df, x =df.index, y=["perf"])

    col2,col3 = st.columns(2)

    loss_container = col2.empty()
    perf_container = col3.empty()

    #loss_container.plotly_chart(loss_fig,use_container_width=True)
    #perf_container.plotly_chart(perf_fig,use_container_width=True)

    col_df,col_predict = st.columns(2)
    df_container = col_df.empty()
    df_container.write(df)
    predict_csv = col_predict.file_uploader("Predict")
    if predict_csv != None:
        dataframe = pd.read_csv(predict_csv,header = None)
        st.write(dataframe)
        smiles = dataframe.iloc[:,0]
        df = pd.DataFrame({"smiles":smiles})
        X = featurizer.featurize(smiles)
        dataset = dc.data.NumpyDataset(X=X)
        mode = task_type
        model = modelclass(
          n_tasks = task_num,
          batch_size=4,
          mode=mode,
          model_dir = "./tmp"
         )

        model.restore()

        output = model.predict(dataset)
    
        n = len(output)
        ntask = task_num
        if mode =="regression":
            if ntask == 1:
                cols = ["prop"]
            else:   
                cols = ["prop_"+str(i) for i in range(ntask)]
            result = pd.DataFrame(output,columns = cols)
            st.write(result)
        else:
            result = np.argmax(output,axis = -1)
            cols = ["prop"+str(i) for i in range(ntask)]
            result = pd.DataFrame(result,columns = cols)
            st.write(result)
        df = pd.concat([df,result],axis=1)
        st.download_button(
        label="Download data as CSV",
        data=df.to_csv(index = False).encode('utf-8'),
        file_name='property.csv',
        mime='text/csv',
    )

    


    best_perf = perf
    best_epoch = 1
    bad_counter = 0

    if(data_type=="classification"):
        bestp=0
    else:
        bestp=10000.0
    bestep=-1

    for i in range(1,epoch):
        loss = model.fit(train_dataset)
        perf = model.evaluate(valid_dataset,[metric])[metric_key]

        new_data = {"epoch":i,"loss":loss,"performance":perf}
        df = df.append(new_data,ignore_index=True)
        
        df_container.write(df)
        #df_all = pd.concat([df_all, df_new], axis=0)
        loss_fig = px.line(df, x=df.index, y=["loss"])
        loss_container.plotly_chart(loss_fig,use_container_width=True)

        perf_fig = px.line(df, x=df.index, y=["performance"])
        perf_container.plotly_chart(perf_fig,use_container_width=True)


        if task_type == 'classification' and perf > best_perf:
            best_perf = perf
            best_epoch = i
            bad_counter = 0
            
        elif task_type == 'regression' and perf < best_perf:
            best_perf = perf
            best_epoch = i
            bad_counter =0
        else :
            bad_counter+=1
        
        #print(bad_counter)
        
        if bad_counter>500:
            break
        
    if test_dataset:
        score = model.evaluate(test_dataset,[metric])[metric_key]
        st.write("Test score",score)

def intro():
    import streamlit as st
    import pandas as pd
    import altair as alt

    st.title("Introduction")

    st.header("Data Visualization")
    st.markdown("Paste a smiles of molecule or upload a csv file")

    st.header("Model Training")
    st.markdown("Train models to predict")

    st.header("Prediction")
    st.markdown("upload csv file to predict properties of molecules")

    st.header("Model Performance")
    st.markdown("Performance on Datasets")
    


page_names_to_funcs = {
    #"Test":test,
    "Introduction": intro,
    "Data Visualization": data_visualization,
    "Model Training": training,
    "Prediction":predict,
    "Performance": performance,
}

demo_name = st.sidebar.selectbox("Choose the Page", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()