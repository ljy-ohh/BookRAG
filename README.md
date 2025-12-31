# BookRAG
完善BookRAG的一些缺陷，并且支持在Windows docker desktop 和 服务器部署

如果有朋友想尝试一下的话，需要修改的文件有：config/pdf_vanilla_siliconflow.yaml,Scripts/cfg/test.yaml,testdemo/testdemo.json
其中pdf_vanilla_siliconflow.yaml需要修改的参数有：api_key，minerU服务我是本地部署的。（当前的api_key已经失效）
Scripts/cfg/test.yaml,testdemo/testdemo.json把想要测试的文件路径填进去

目前还在debug阶段，还没有成功

2025-12-31 更新
目前在Windows本地测试通过了index和rag的测试，命令分别如下
#  index
python main.py -c config\pdf_vanilla_siliconflow.yaml -d Scripts\cfg\test.yaml --nsplit 1 --num 1 index --stage all
#  rag
python main.py -c config\pdf_vanilla_siliconflow.yaml -d Scripts\cfg\test.yaml --nsplit 1 --num 1 rag


