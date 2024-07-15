CALL_DIR=$PWD
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. "${SCRIPT_DIR}/tools.sh"

# Path to Spot workspace, relative to repository root;
# No begin or trailing slash.
SPOT_PATH=${SCRIPT_DIR}


# Set the ID of the Spot you are working on. Either
# 12 (stands for 12070012) or 2 (stands for 12210002)
SPOT_ID="2"
if [ $SPOT_ID != "12" ] && [ $SPOT_ID != "2" ]; then
    echo "Invalid SPOT ID. Either 12 (stands for 12070012) or 2 (stands for 12210002)."
    return 1
fi

# Configure the IP addresses for different network connections
SPOT_ETH_IP="10.0.0.3"
SPOT_WIFI_IP="192.168.80.3"


#------------- FUNCTIONS  ----------------
# Always assume at the start of a function,
# or any if clause, the working directory is
# the root directory of the repository.
# Detect your Spot connection.
function detect_spot_connection
{
    # Detects the spot connection by pinging.
    # Sets two variables, 'spot_conn' and 'spot_ip'
    echo -e "Pinging Spot WiFi IP $SPOT_WIFI_IP..."
    if ping_success $SPOT_WIFI_IP; then
        echo -e "OK"
        spot_conn="spot wifi"
        spot_ip=$SPOT_WIFI_IP
        true && return
    fi

    echo -e "Pinging Spot Ethernet IP $SPOT_ETH_IP..."
    if ping_success $SPOT_ETH_IP; then
        echo -e "OK"
        spot_conn="ethernet"
        spot_ip=$SPOT_ETH_IP
        true && return
    fi


    echo "Cannot connect to Spot"
    spot_conn=""
    spot_ip=""
    false
}


function ping_spot
{
    if [ -z $SPOT_IP ]; then
        echo -e "It appears that Spot is not connected"
    else
        ping $SPOT_IP
    fi
}


# Add a few alias for pinging spot.
#------------- Main Logic  ----------------

# We have only tested Spot stack with Ubuntu 20.04.
if ! ubuntu_version_equal 20.04; then
    echo "SPOT development requires Ubuntu 20.04. Abort."
    return 1
fi


# create a dedicated virtualenv for spot workspace
if [ ! -d "${SPOT_PATH}/venv/spot" ]; then
    echo "SPOT connection should be within the virtual environment"
    return 1
fi

# source ${SPOT_PATH}/venv/spot/bin/activate



export PYTHONPATH=""

# We'd like to use packages in the virtualenv
export PYTHONPATH="${SPOT_PATH}/venv/spot/lib/python3.8/site-packages:${PYTHONPATH}:/usr/lib/python3/dist-packages"
if confirm "Are you working on the real robot ?"; then
    # Check if the environment variable SPOT_IP is set.
    # If not, then try to detect spot connection and set it.
    if [ -z $SPOT_IP ]; then
       if detect_spot_connection; then
           export SPOT_IP=${spot_ip}
           export SPOT_CONN=${spot_conn}
       fi
    fi

    # If Spot is connected, then SPOT_IP should be set.
    if [ -z $SPOT_IP ]; then
        echo -e "Unable to connect to spot."
    else
        if ping_success $SPOT_IP; then
            echo -e "Spot connected! IP: ${SPOT_IP}; Method: ${SPOT_CONN}"
        else
            echo -e "Spot connection lost."
            export SPOT_IP=""
            export SPOT_CONN=""
        fi
    fi

    # Load the spot passwords
    source $SPOT_PATH/.spot_passwd
fi

export SPOT_ARM=1

cd $CALL_DIR
echo $SPOT_PATH