#!/bin/bash

# Define options
options=("Download Chat Model" "Download Code Model" "Set Environment" "Deploy Chat Server" "Deploy Code Server" "Done")

# Array to hold the state (selected/not selected) of each option
selected=(false false false false)

# Current cursor position
current=0

# Function to display the menu
show_menu() {
    # Clear screen
    echo -en "\e[1;1H\e[2J"
    echo "Luacoder endpoint server deploy script"
    echo "Use arrow keys to navigate and enter/space to toggle options."
    echo "Press enter to confirm:"
    for i in "${!options[@]}"; do
        # Highlight the current cursor position
        if [ "$i" -eq "$current" ]; then
            echo -en "\e[7m"  # Start highlight
        fi

        # Display checkbox, but not for "Done"
        if [ "$i" -ne $((${#options[@]}-1)) ]; then
            if [ "${selected[$i]}" = true ]; then
                echo -n "[x] ${options[$i]}"
            else
                echo -n "[ ] ${options[$i]}"
            fi
        else
            echo -n "    ${options[$i]}"
        fi

        # End highlight and move to next line
        echo -en "\e[0m\n"
    done
}

# Main loop
while true; do
    # Display menu
    show_menu

    # Read user input
    IFS= read -rsn1 input

    # Handle arrow keys, space, and enter
    case $input in
        # Arrow up
        $'\x1b')
            read -rsn2 -t 0.1 input
            case $input in
                '[A')
                    ((current--))
                    if [ "$current" -lt 0 ]; then
                        current=$((${#options[@]}-1))
                    fi
                    ;;
                # Arrow down
                '[B')
                    ((current++))
                    if [ "$current" -ge "${#options[@]}" ]; then
                        current=0
                    fi
                    ;;
            esac
            ;;
        # Space or Enter (to toggle checkbox)
        " " | "")
            # If "Done" is selected, break out of the loop
            if [ "$current" -eq $((${#options[@]}-1)) ]; then
                break
            fi
            selected[$current]=$([ "${selected[$current]}" = true ] && echo false || echo true)
            ;;
    esac
done

echo -en "\e[1;1H\e[2J"
echo "##############################"
echo "Deploying luacoder-endpoint-server"
# Display selected options
echo "You selected:"
for i in "${!options[@]}"; do
    if [ "${selected[$i]}" = true ]; then
        echo "${options[$i]}"
    fi
done

echo "##############################"
echo "check environment, make sure python=3.10 cuda=11.7"
python_version=$(python --version 2>&1)
echo "Python version: $python_version"

# Check CUDA version using nvcc
cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
if [ -z "$cuda_version" ]; then
  echo "CUDA is not installed!"
else
  echo "CUDA version: $cuda_version"
fi

# Download chat model section
if [ "${selected[0]}" = true ]; then
    echo "##############################"
    echo "clean files"
    rm -rf ./models/luachat/*
    echo "##############################"
    echo "Download luachat model"
    wget -r -nd -np -q --show-progress --progress=bar ftp://10.246.52.90/model/luachat-7b/ -P ./models/luachat
fi


# Download code model section
if [ "${selected[1]}" = true ]; then
    echo "##############################"
    echo "clean files"
    rm -rf ./models/luacoder/*
    echo "##############################"
    echo "Download luacoder model"
    wget -r -nd -np -q --show-progress --progress=bar ftp://10.246.52.90/model/luacoder-7b/ -P ./models/luacoder 
fi


# Set environment section
if [ "${selected[2]}" = true ]; then
    echo "##############################"
    echo "Install vllm and its dependencies"
    pip install vllm
    echo "##############################"
    echo "Install fastapi and its dependencies"
    pip install -r requirements.txt
fi

# Deploy chat server section
if [ "${selected[3]}" = true ]; then
    # Deploy section
    echo "##############################"
    echo "Deploy server endpoint, using cuda:0"
    CUDA_VISIBLE_DEVICES=0 python src/chat_server.py --host=0.0.0.0 --port=7088
fi

# Deploy code server section
if [ "${selected[4]}" = true ]; then
    # Deploy section
    echo "##############################"
    echo "Deploy server endpoint, using cuda:0"
    CUDA_VISIBLE_DEVICES=0 python src/code_server.py --host=0.0.0.0 --port=10000
fi