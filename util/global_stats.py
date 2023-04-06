"""store testing performance vector, and use it as a global variable"""

def _init():#初始化
    global _global_dict
    _global_dict = {}
 
def set_value(key,value):
    """ define a global variable """
    _global_dict[key] = value
 
def get_value(key, defValue=None):
    """ get the value of a global variable, return default value if not exist """
    try:
        return _global_dict[key]
    except KeyError:
        return defValue

