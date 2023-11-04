# VSCode extension for DevAssistantChat and DevAssistant

It was forked from [tabnine-vscode](https://github.com/codota/tabnine-vscode) & modified for making it compatible with open source code models. 

## 版本
**DevAssistant-vscode beta 0.1.1**

- Starting a conversation
- Sidebar can be clicked to start a dialog
- Improve code collection function

**DevAssistant-vscode beta 0.1.0**

- Launch code-completion feature preview, model will grow iteratively
- DevAssistant service control function
- Disable DevAssistantChat interface
- This plugin will conflict with vscode copilot, please choose one to enable.
- This plugin will collect user's code and access user's hard disk.
- Currently the generated model is not perfect, if you don't like the trouble, you can click to close it.

**DevAssistant-vscode beta 0.0.2**

- Update server nodes
- Setting config configuration bug fixes


**DevAssistant-vscode beta 0.0.1**

- Implement DevAssistantChat interface, DevAssistantChat api
- Not enable code auto-completion yet, waiting for model optimization.
- The model has about 20% error rate
- Please don't feed too many requests at once, DevAssistantChat average reply time is 2 seconds.



## How to install

Method 1: Click Install from VSIX in the upper right corner of Extension.

<img src="https://img-blog.csdnimg.cn/20201217145528782.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ZpZ2h0c3lq,size_16,color_FFFFFF,t_70">

Install success as below

<img src="https://img-blog.csdnimg.cn/20201217145616885.png">

Method 2: Ctrl + Shift + P -> Type Install from... -> Select Extensions:Install from VSIX.... -> Select Extensions:Install from VSIX... Open the file selection
The process is the same as above!

Method 3:
1. Bring up the Terminal command line viewport;
2. enter the command: code --install-extension xxx.vsix (this vsix file path is recommended to drag and drop);
3. execute the command, the installation is complete we will see the message:
4. restart VS Code can be seen in Extensions just installed plug-ins!


## How to use

After deploying server and vscode extension installation you can see DevAssistant in the status bar and DevAssistantChat in the sidebar.

<img src="https://github.com/Miraclove/images/blob/main/dev%20assistant/overview.png?raw=true">


Click on the DevAssistant icon to start or pause the code-completion service.

<img src="https://github.com/Miraclove/images/blob/main/dev%20assistant/disable.png?raw=true">



## How to develop

1. Install the dependency: yarn install --frozen-lockfile
2. In VSCode, open the Run and Debug sidebar and click Launch Extension.
3. Compile and build: vsce package

## Related

| Repository | Description |
| --- | --- |
| [huggingface-vscode](https://github.com/huggingface/llm-vscode) | Code autocomplete |
