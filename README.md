# DevAssistant and its applications

DevAssistant is an innovative tool designed to streamline and enhance the software development process by leveraging a pre-trained 7-billion parameter language model. This repository showcases the development and deployment of an LLM specifically fine-tuned to assist developers in writing code that seamlessly integrates with a company's API documentation.

## Key Features:

- Custom Model Training: The core of DevAssistant is a PyTorch-based LLM, pretrained on a comprehensive corpus of Lua code and company-specific API documentation, ensuring high relevance and understanding of domain-specific language.
- Fine-Tuning for Instructional Interaction: The model is fine-tuned to not just complete code but also to facilitate instructional dialogue, helping developers understand codebase and APIs more effectively.
- Data Handling: A systematic approach to data collection, cleaning, and modification was employed to prepare the dataset for LLM training, adhering to strict quality standards.
- Scalable Training Pipeline: Utilization of Accelerate and DeepSpeed frameworks for an optimized training pipeline, enabling rapid and scalable model training using four A100 GPUs in a Linux environment.
- Model Evaluation: Development of a unique evaluation script to measure the model's performance by comparing the generated code and responses to the actual API documentation, focusing on practical utility and accuracy.
- Endpoint Server for Model Serving: Setup of a VLLM-powered endpoint server to serve the model, ensuring fast and reliable inference for real-time applications.
- VSCode Plugin: Creation of a TypeScript-based VSCode plugin to provide in-line code completion, drawing on the capabilities of the pretrained large language model. This feature aims to enhance developer productivity by suggesting accurate and context-aware code snippets. Interactive Sidebar Chat Interface: An additional sidebar in the VSCode plugin allows developers to engage in a chat with the fine-tuned model, obtaining on-the-fly assistance and clarification, further enriching the development environment.

## Preview

After deploying server and vscode extension installation you can see DevAssistant in the status bar and DevAssistantChat in the sidebar.

<img src="https://github.com/Miraclove/images/blob/main/dev%20assistant/overview.png?raw=true">


Click on the DevAssistant icon to start or pause the code-completion service.

<img src="https://github.com/Miraclove/images/blob/main/dev%20assistant/disable.png?raw=true">



## Version
DevAssistant kit beta 0.0.1


**DevAssistant-model 0.0.1** - data processing, model training and fine-tuning (research and development)
- data processing for chat interaction and code generation
- model training using deepspeed with chat and code generation model
- evaluation with code and chat inference


**DevAssistant-vscode beta 0.1.1** - the vscode plugin for code autocomplete and chat (deployed at client side)

- Starting a conversation
- Sidebar can be clicked to start a dialog
- Improve code collection function


**DevAssistant-endpoint-server beta 0.1.0** - the server providing api for chat and text generation (deployed at server side)

- auto deploy scripts for pulling model
- data collection enable


**Train-Flow beta 0.0.1** - train model management platform
- use command to execute scripts in local disk
- use web interface to diaplay current running scripts and other information

**for detailed information, please reach to each subfolder**