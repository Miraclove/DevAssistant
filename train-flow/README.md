# Train-Flow: train model management platform

Train-Flow which is a webserver that automatic handle multiple training scripts in queue and execute one by one. Currently, it supports real time output display.

**Very basic, easy to use.**


## Version

**train-flow beta 0.0.1**

- use command to execute scripts in local disk
- use web interface to diaplay current running scripts and other information


## File Structure

```
train-flow/
    templates/             # the html webpage
    .gitignore          # git ignore some file
    LICENSE
    README.md
    requirements.txt    # pip requirement
    test.py             # test scripts
    main.py             # webserver backend
```



## Usage

```shell
pip install -r requiremnets.txt
python main.py
```

Then use http://localhost:5000 to access the web pages and add command to queue.