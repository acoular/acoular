.. tab-set::
    :sync-group: tool

    .. tab-item:: ``uv``
        :sync: uv

        .. code-block:: console

            $ uv venv

        .. note::
           ``uv`` will handle environment activation implicitly (the environment is created at ``.venv``).

    .. tab-item:: ``venv``
        :sync: pip

        .. code-block:: console

            $ python3 -m venv my-env

        and activate the environment with:

        .. code-block:: console

            $ source my-env/bin/activate

    .. tab-item:: ``mamba``
        :sync: mamba

        .. code-block:: console

            $ mamba create -n my-env

        and activate the environment with:

        .. code-block:: console

            $ mamba activate my-env

    .. tab-item:: ``conda``
        :sync: conda

        .. code-block:: console

            $ conda create -n my-env

        and activate the environment with:

        .. code-block:: console

            $ conda activate my-env
