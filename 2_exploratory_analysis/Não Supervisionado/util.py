"Script que busca a localização do módulo customizado."

from pathlib import Path
import sys
import glob

def init()-> str:
    """Informa ao usuário que o módulo foi importado com sucesso.

    Returns
    -------
    str
        Mensagem de importação.
    """
    module_name_1 = Path(__file__).stem
    return f"Módulo {module_name_1} importado com sucesso."

def search_module(module_name_2: str = "util") -> Path:
    """Responsável por procurar o arquivo `__base{module_name_2}__`.
    Este arquivo diz onde estão os arquivos principais do projeto
    que incluem a localização de todos os arquivos.

    Parameters
    ----------
    module_name_2 : str
        Nome do módulo a ser importado.

    Returns
    -------
    Path
        Caminho com a localização dos módulos.
    """
    path_search = Path(__file__).parent
    was_found = False
    while not was_found:
        path_search = path_search.parent
        path_config_file = glob.glob(str(path_search) + f"/**/__base{module_name_2}__",
                                     recursive = True)
        len_config_file = len(path_config_file)
        if (len_config_file != 0) and \
           (not len_config_file > 1) and \
           (f'__base{module_name_2}__'  in path_config_file[0]):
            return Path(path_config_file[0]).parent
    raise f"Module {module_name_2} not found."

module_name = Path(__file__).stem 
path_module = search_module(module_name)
sys.path.append(str(path_module))


if __name__ == '__main__':
    pass
