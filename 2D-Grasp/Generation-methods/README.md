# HOW TO USE

The code is integrated from [GRCNN](#https://github.com/skumra/robotic-grasping) and [TF-Grasp](#https://github.com/WangShaoSUN/grasp-transformer)

**If there are any misunderstandings, please leave and communicate with me or review the original project code**

## Environment
This code was developed with Python 3.6 on Ubuntu 16.04.  Python requirements can installed by:

```bash
pip install -r requirements.txt
```

## Datasets


Currently, this code support Cornell and Jacquard Dataset. 
About these tow dataset ,you can see the homepage [there](https://github.com/YangXuanyi/grasp-encyclopedia#Datasets)


## Training

Training is done by the `main.py` script.  

Standard commands:

```
python main.py   --method your method --dataset your dataset
```

Some basic examples:

```bash
# Train  on Cornell Dataset and use TF-Grasp method
python main.py   --method TF-Grasp --dataset cornell

# Train  on Jacquard Dataset and use GRCNN method
python main.py   --method GRCNN --dataset Jacquard
```

训练后的模型将会在`.output/model/`中保存，具体保存文件的类型在main程序中设置。

## Visualize

At present, there is no integration of args hyperparameter files in this section, so it is necessary to change the parameters needed one by one according to the different testing networks


Some basic examples:
```bash
# visulaize grasp rectangles
python visualise_grasp_rectangle.py   --network your 

# visulaize heatmaps
python visulaize_heatmaps.py  --network your network path
```

## 注意事项！
* tain和visual程序的超参数都集成在了opts文件中，本项目目前没有对这些参数进行微调和优化，也就是说这些参数设置极大可能不是最优的。在使用该项目时一定要根据自己的需求重新设置这些超参数，否则无法达到理想效果。
* network的具体地址为`output/models/`
* 因为main程序训练后保存的是模型的全部信息，所以在测试时直接使用`torch.load`方法就可以完整的导入模型，不需要实例化要使用的模型，因此在可视化的opts参数中不需要特殊区分method的，导入哪种模型文件，就是哪种网络。
* 现在models文件夹內有一个用几张cornell图像训练1个epoch的GRCNN模型，可以直接调用它来测试可视化程序。这个模型的acc=0，不准备任何实际意义。用它可以在不训练模型的情况下提前查看可视化程序运行后应用的效果。