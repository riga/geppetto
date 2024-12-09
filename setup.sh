#!/usr/bin/env bash

setup_geppetto() {
    #
    # prepare local variables
    #

    local shell_is_zsh=$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"
    local orig="${PWD}"

    if ${shell_is_zsh}; then
        emulate -L bash
        setopt globdots
    fi

    #
    # global variables
    # (GP = GePpetto)
    #

    # start exporting variables
    export GP_BASE="${this_dir}"
    export GP_DATA_BASE="${GP_DATA_BASE:-/data/dust/cms/user/$( whoami )/geppetto_data}"
    export GP_VENV_BASE="${GP_VENV_BASE:-${GP_DATA_BASE}/venvs}"

    # external variables
    export LANGUAGE="${LANGUAGE:-en_US.UTF-8}"
    export LANG="${LANG:-en_US.UTF-8}"
    export LC_ALL="${LC_ALL:-en_US.UTF-8}"
    export PYTHONWARNINGS="ignore"
    export VIRTUAL_ENV_DISABLE_PROMPT="${VIRTUAL_ENV_DISABLE_PROMPT:-1}"
    export HF_HOME="${HF_HOME:-${GP_DATA_BASE}/hf_datasets}"
    export SPACY_DATA_DIR="${SPACY_DATA_DIR:-${GP_DATA_BASE}/spacy}"

    #
    # minimal local software setup
    #

    export PYTHONPATH="${GP_BASE}:${PYTHONPATH}"

    ulimit -s unlimited

    # remove software stack if requested
    if [ "${GP_REINSTALL_VENV}" = "1" ]; then
        echo "removing venvs at ${GP_VENV_BASE}"
        rm -rf "${GP_VENV_BASE}"
    fi

    local venv_existing="$( [ -d "${GP_VENV_BASE}" ] && echo "true" || echo "false" )"
    if ! ${venv_existing}; then
        # setup the venv
        echo "setting up venv at ${GP_VENV_BASE}"
        mkdir -p "${GP_VENV_BASE}"
        python3 -m venv "${GP_VENV_BASE}/gpt" || return "$?"
    fi

    # activate it
    source "${GP_VENV_BASE}/gpt/bin/activate" "" || return "$?"

    if ! ${venv_existing}; then
        # install requirements
        pip install -U pip setuptools wheel || return "$?"
        pip install -r "${GP_BASE}/requirements.txt" || return "$?"

        # manually install spacy language packages
        python -m spacy download en_core_web_sm -t "${SPACY_DATA_DIR}"
    fi

    # remember the full english language model path
    export GP_SPACY_MODEL_EN="${SPACY_DATA_DIR}/en_core_web_sm/$( ls -1 "${SPACY_DATA_DIR}/en_core_web_sm" | grep en_core_web_sm | head -n 1 )"
}

setup_geppetto "$@"
