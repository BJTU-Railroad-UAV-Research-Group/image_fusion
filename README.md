## 0.环境配置

创建虚拟环境；

激活虚拟环境

安装第三方库

```cmd
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirement.txt
```



## 1.抠图样本存放于文件夹

使用`sample_aug`工程生成的`AugSamples`置于当前工程的根目录



## 2.修改配置文件

按需修改`config/config.yml`



## 3.执行增强主程序

```python
python main.py
```



## 4.查看结果

查看输出文件夹`output`下的结果