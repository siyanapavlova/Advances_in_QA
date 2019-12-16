#!/usr/bin/python3.6

"""
This module provides helper functions.
"""

import sys
import re
#from tqdm import tqdm
#import time


def loop_input(rtype=str, default=None, msg=""):
    """
    Wrapper function for command-line input that specifies an input type
    and a default value.
    :param rtype: type of the input. one of str, int, float, bool
    :type rtype: type
    :param default: value to be returned if the input is empty
    :param msg: message that is printed as prompt
    :type msg: str
    :return: value of the specified type
    """
    while True:
        try:
            s = input(msg+f" (default: {default}): ")
            if rtype == bool and len(s) > 0:
                if s=="True":
                    return True
                elif s=="False":
                    return False
                else:
                    print("Input needs to be convertable to",rtype,"-- try again.")
                    continue
            else:
                return rtype(s) if len(s) > 0 else default
        except ValueError:
            print("Input needs to be convertable to",rtype,"-- try again.")
            continue


class ConfigReader():
    """
    Basic container and management of parameter configurations.
    Read a config file (typically ending with .cfg), use this as container for
    the parameters during runtime, and change/write parameters.

    CONFIGURATION FILE SYNTAX
    - one parameter per line, containing a name and a value
        - name and value are separated by at least one white space or tab
        - names should contain alphanumeric symbols and '_' (no '-', please!)
    - list-like values are allowed (use Python list syntax)
        - strings within valuelists don't need to be quoted
        - value lists either with or without quotation (no ["foo", 3, "bar"] )
        - mixed lists will exclude non-quoted elements
    - multi-word expressions are marked with single or double quotation marks
    - lines starting with '#' are ignored
    - no in-line comments!
    - config files should have the extension 'cfg' (to indicate their purpose)
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.params = self.read_config()

    def __repr__(self):
        """
        returns tab-separated key-value pairs (one pair per line)
        """
        return "\n".join([str(k)+"\t"+str(v) for k,v in self.params.items()])

    def read_config(self):
        #TODO docstring
        cfg = {}
        with open(self.filepath, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.rstrip()
            if not line: # ignore empty lines
                continue
            elif line.startswith('#'): # ignore comment lines
                continue

            words = line.split()
            paramname = words.pop(0)
            if not words: # no value specified
                print(f"WARNING: no value specified for parameter {paramname}.")
                paramvalue = None
            elif words[0].startswith("["): # detects a list of values
                paramvalue = self.listparse(" ".join(words))
            elif words[0].startswith('"') or words[0].startswith('\''): # detects a multi-word string
                paramvalue = self.stringparse(words)
            else:
                """ only single values are valid! """
                if len(words) > 1:
                    #TODO make this proper error handling (= throw an error)
                    print(f"ERROR while parsing {self.filepath} --",
                          f"too many values in line '{line}'.")
                    sys.exit()
                else:
                    """ parse the single value """
                    paramvalue = self.numberparse(words[0]) # 'words' is still a list
                    paramvalue = self.boolparse(paramvalue)

            cfg[paramname] = paramvalue # adds the parameter to the config

        self.config = cfg
        return self.config

    @classmethod
    def listparse(cls, liststring):
        """
        #TODO docstring
        """
        re_quoted = re.compile('["\'](.+?)["\'][,\]]')
        elements = re.findall(re_quoted, liststring)
        if elements:
            return elements

        re_unquoted = re.compile('[\[\s]*(.+?)[,\]]')
        elements = re.findall(re_unquoted, liststring)
        if elements:
            result = []
            for e in elements:
                e = cls.numberparse(e)  # convert to number if possible
                e = cls.boolparse(e)  # convert to bool if possible
                result.append(e)
            return result

    @staticmethod
    def stringparse(words):
        words[0] = words[0][1:] # delete opening quotation marks
        words[-1] = words[-1][:-1] #delete closing quotation marks
        return " ".join(words)

    @staticmethod
    def numberparse(string):
        """
        Tries to convert 'string' to a float or even int.
        Returns int/float if successful, or else the input string.
        """
        try:
            floaty = float(string)
            if int(floaty) == floaty:
                return int(floaty)
            else:
                return floaty
        except ValueError:
            return string

    @staticmethod
    def boolparse(string):
        if string == "True" or string == "False":
            return bool(string)
        else:
            return string

    def get_config(self):
        """
        returns the config as a dictionary
        """
        return self.params

    def get_params(self):
        """
        returns a list of parameter names
        """
        return [key for key in self.params.keys()]

    def get(self, *paramname):
        """
        returns a specific value or a tuple of values corresponding to the
        provided parameter name(s).
        """
        values = [self.params[name] for name in paramname]
        if len(values) == 1:
            return values[0]
        else:
            return tuple(values)

    def set(self, paramname, value):
        self.params.update({paramname:value})