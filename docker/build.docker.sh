# Make sure we are in the right directory.
# the rest of the script assumes we are in robotdev/docker
if [[ $PWD = *multi-purpose-representation ]]; then
    cd docker
elif [[ ! $PWD = *multi-purpose-representation/docker ]]; then
    echo -e "You must be in either 'multi-purpose-representation' or the sub-directory 'docker' to run this command."
    return 1
fi

# load tools (helper functions to build up the docker image & connect to SPOT)
. "../tools.sh"

# parse args

# UID/GID for the container user
hostuser=$USER
hostuid=$UID
hostgroup=$(id -gn $hostuser)
hostgid=$(id -g $hostuser)
# allows user to supply a custom suffix
custom_tag_suffix="default"
# always provide GPU support to the container
nvidia=".nvidia"
for arg in "$@"
do
    if parse_var_arg $arg; then
        if [[ $var_name = "hostuser" ]]; then
            hostuser=$var_value
        elif [[ $var_name = "tag-suffix" ]]; then
            custom_tag_suffix="-$var_value"
        else
            echo -e "Unrecognized argument variable: ${var_name}"
        fi
    fi
done

# Build the docker image.  The `--rm` option is for you to more conveniently
# rebuild the image.
cd $PWD/../  # get to the root of the repository
docker build -f ./docker/Dockerfile${nvidia}\
       -t multi-purpose-representation:$custom_tag_suffix\
       --build-arg hostuser=$hostuser\
       --build-arg hostgroup=$hostgroup\
       --build-arg hostuid=$hostuid\
       --build-arg hostgid=$hostgid\
       --rm\
       .
# Explain the options above:
# -t: tag
