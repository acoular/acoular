.. tab-set::
    :sync-group: tool

    .. tab-item:: ``uv``
        :sync: uv

        .. note::

           ``uv`` defaults to an editable installation, so nothing needs to be done here.

    .. tab-item:: ``pip``
        :sync: pip

        Make sure to use an up-to-date version of ``pip`` that supports ``dependency-groups``. You can update ``pip`` with:

        .. code-block:: console

            $ pip install --upgrade pip

        Then, make an editable installation of Acoular and its ``dev`` dependencies with:

        .. code-block:: console

            $ pip install -Ue .'[full]' --group dev

    .. tab-item:: ``mamba``
        :sync: mamba

        Install ``pip`` with:

        .. code-block:: console

            $ mamba install pip

        Then, make an editable installation of Acoular and its ``dev`` dependencies with:

        .. code-block:: console

            $ pip install -Ue .'[full]' --group dev

    .. tab-item:: ``conda``
        :sync: conda

        Install ``pip`` with:

        .. code-block:: console

            $ conda install pip

        Then, make an editable installation of Acoular and its ``dev`` dependencies with:

        .. code-block:: console

            $ pip install -Ue .'[full]' --group dev
