from  utils import load
from utils import parallel, save
import javalang
import re
from tqdm import tqdm
from collections import deque
from functools import partial
import re
from pythonds.basic.stack import Stack

node_maxlen = 15
max_nodes = 400 
max_tokens = 450
# max_statms = 60
max_statms = 40
min_statms = 2
subtoken = True
workers = 4
controlList = ['if','else','else if','while','do','switch','case','break','continue','for','try','catch']



def parseData(codelines):
    return removeSpace(codelines)

def removeSpace(codelines):
    lines = list()
    lines.clear()  # To clear the initial Lines list
    for line in codelines:
        if len(line) == 0:
            continue

        lines.append(re.sub(r'\s+', ' ', line).strip())

    return CheckLines(lines)


def CheckLines(lines):
    stack = Stack()

    num = Stack()
    breakFlags = Stack()
    continueFlags = Stack()
    ifFlag = list()

    line_num = 0
    endifFlag = Stack()
    #endelse = 0
    #endelif = Stack()
    size = len(lines)
    endtry =0
    #endcatch =0
    matrix = [[0 for x in range(size)] for x in range(size)]

    while (line_num < size):
        if "#" in lines[line_num]:
            matrix[line_num][line_num + 1] = 1

        elif re.match('if',lines[line_num].strip()) and lines[line_num].strip()[-1]!=';':
            matrix[line_num][line_num + 1] = 1
            stack.push('if')
            num.push(line_num)
            ifFlag.append(line_num)

        elif re.match('else if',lines[line_num].strip()) and lines[line_num].strip()[-1]!=';':
            matrix[line_num][line_num + 1] = 1
            #matrix[endif][line_num] =1
            stack.push('else if')
            num.push(line_num)
            ifFlag.append(line_num)

        elif re.match('else',lines[line_num].strip()) and lines[line_num].strip()[-1]!=';':
            matrix[line_num][line_num + 1] = 1
            #matrix[num.peek()][line_num] = 1
            while 0!=len(ifFlag):
                ifnum = ifFlag.pop()
                for i in range(len(ifFlag)):
                    matrix[ifFlag[i]][ifnum] =1
                matrix[ifnum][line_num]=1
            stack.push('else')
            num.push(line_num)

        elif re.match('while ',lines[line_num].strip()) and lines[line_num].strip()[-1]!=';':  # 不是 do while
            matrix[line_num][line_num + 1] = 1
            stack.push('while')
            num.push(line_num)

        elif re.match('do',lines[line_num].strip()) and lines[line_num].strip()[-1]!=';':
            matrix[line_num][line_num + 1] = 1
            stack.push('do')
            num.push(line_num)

        elif re.match('for',lines[line_num].strip()) and lines[line_num].strip()[-1]!=';':
            matrix[line_num][line_num + 1] = 1
            stack.push('for')
            num.push(line_num)

        elif re.match('switch',lines[line_num].strip()) and lines[line_num].strip()[-1]!=';':
            matrix[line_num][line_num + 1] = 1
            stack.push('switch')
            num.push(line_num)

        elif re.match('case',lines[line_num].strip()) and lines[line_num].strip()[-1]!=';':
            matrix[line_num][line_num + 1] = 1
            matrix[num.peek()][line_num] = 1

        elif re.match('default: ',lines[line_num].strip()) and lines[line_num].strip()[-1]!=';':
            matrix[line_num][line_num + 1] = 1
            matrix[num.peek()][line_num] = 1

        elif re.match('try',lines[line_num].strip()) and lines[line_num].strip()[-1]!=';':
            matrix[line_num][line_num + 1] = 1
            stack.push('try')
            num.push(line_num)
        elif re.match('catch',lines[line_num].strip()) and lines[line_num].strip()[-1]!=';':
            #matrix[line_num][line_num + 1] = 1
            stack.push('catch')
            num.push(line_num)
            #matrix[num.peek()][line_num] = 1
        elif re.match('synchronized',lines[line_num].strip()) and lines[line_num].strip()[-1]!=';':
            matrix[line_num][line_num + 1] = 1
            stack.push('synchronized')
            num.push(line_num)


        elif "break" in lines[line_num]:
            breakFlags.push(line_num)

        elif "continue" in lines[line_num]:
            continueFlags.push(line_num)

        elif "{" in lines[line_num]:
            matrix[line_num][line_num + 1] = 1


        elif "}" in lines[line_num]:

            if line_num == size - 1:  # 最后一行
                break
            if 0 == stack.size():
                line_num += 1
                continue
            #matrix[line_num][line_num + 1] = 1
            s = stack.peek()

            if breakFlags.isEmpty() == False:
                if s is not 'if' and s is not 'else' and s is not 'else if':
                    matrix[breakFlags.pop()][num.peek()] = 1

            if continueFlags.isEmpty() == False:
                if s is not 'if' and s is not 'else' and s is not 'else if':
                    matrix[continueFlags.pop()][num.peek()] = 1



            if s == 'if' or s == 'else if':
                #endif = line_num
                endifFlag.push(line_num)
                if len(ifFlag)!=0:
                    matrix[ifFlag[-1]][line_num+1] =1

            if s == 'else':
                endelse = line_num
                #matrix[endif][endelse+1] = 1
                while(0!=endifFlag.size()):
                    enifnum = endifFlag.pop()
                    matrix[enifnum][line_num+1] =1
            if s == 'try':
                endtry = line_num
            if s == 'catch':
                endcatch = line_num
                matrix[endtry][endcatch+1] = 1

            if s == 'while'or s=='for':

                matrix[line_num][num.peek()] =1
            if s == 'do':
                matrix[line_num+1][num.peek()+1] = 1

            if s is not 'switch' and num.size()!=0:
                stack.pop()
                num.pop()
            if s is not 'if' or s is not 'try':
                matrix[line_num][line_num + 1] = 1


        elif line_num + 1 != size:
            matrix[line_num][line_num + 1] = 1

        line_num += 1
    return matrix


def printMatrix(matrix):
    print("PRINTING...")
    out = open("output_while.txt", "w")
    i = 1
    size = len(matrix)

    for k in range(1, size + 1):
        print(k, end=" ")

    print()

    for row in matrix:
        out.write(str(row) + "\n")

        for j in row:
            print(j, end=" ")
        print(i)
        i += 1

    out.close()

def isNode(codeline,api_seqs):
    type = 'NoneNode'
    value = None
    reStr = ''
    apilist=[]
    if 0!=len(api_seqs):
        for api in api_seqs:
            reStr =  api+'|'+reStr
        reStr =reStr[:-1]

        apilist = re.findall(reStr,codeline)
        apilist.reverse()
    if 0!= len(apilist):
        type = 'Method_CALL'
        value = apilist

    for controlUnit in controlList:
        #if controlUnit in codeline:
        if re.match(controlUnit+' ',codeline.strip()):
            type = controlUnit


    return type,value

def getFuncName(code):
    """
    :param code: code snippet;
    :return: function name in input code snippet
    """
    obj = re.search(r'\s?(\w*?)\s?\(',code)
    if None == obj:
        print(code)
        return 'funcName'
    return obj.group(1)

def constructAPIContextGraphNodes(codelines, api_seqs):
    apiContextGraph = []
    nodeID = 0

    newNode = {}
    newNode['id'] = nodeID
    newNode['value'] = codelines[0]
    newNode['type'] = 'Func_Name'
    newNode['api'] = [getFuncName(codelines[0]+codelines[1])]
    newNode['label'] = None
    # newNode['type'] = 'funName'
    nodeID = nodeID + 1
    apiContextGraph.append(newNode)

    for codeline in codelines[1:]:
        nodeResult = isNode(codeline, api_seqs)
        if 'Method_CALL' == nodeResult[0]:
            newNode = {}
            newNode['id'] = nodeID
            newNode['value'] = codeline.strip()
            newNode['api'] = nodeResult[1]
            newNode['type'] = nodeResult[0]
            newNode['label'] = None
            nodeID = nodeID + 1
            apiContextGraph.append(newNode)
        elif 'NoneNode'!=  nodeResult[0]:

            newNode = {}
            newNode['id'] = nodeID
            newNode['value'] = codeline.strip()
            newNode['api'] = nodeResult[1]
            newNode['type'] = nodeResult[0]
            newNode['label'] = None
            nodeID = nodeID + 1
            apiContextGraph.append(newNode)

    return apiContextGraph


def constructAPIContextGraphNodes_(codelines,api_seqs):
    apiContextGraph = []
    nodeID =0

    newNode = {}
    newNode['id'] = nodeID
    newNode['value'] = codelines[0]
    newNode['type'] = 'funName'
    #newNode['type'] = 'funName'
    nodeID = nodeID + 1
    apiContextGraph.append(newNode)

    for codeline in codelines[1:]:
        nodeResult = isNode(codeline,api_seqs)
        if 'Method_CALL' == nodeResult[0]:
            # 考虑嵌套的东西欸
            newNode = {}
            newNode['id'] = nodeID
            newNode['value'] = codeline.strip()
            newNode['type'] = isNode(codeline,api_seqs)
            nodeID = nodeID + 1
            apiContextGraph.append(newNode)

    return apiContextGraph

def delnode(nodeid,matrix):
    children =[]
    for i,item in enumerate(matrix[nodeid]):
        if 1==item:
            children.append(i)
    parents = []
    for i,row in enumerate(matrix):
        if 1 == row[nodeid]:
            parents.append(i)
    for p in parents:
        for c in children:
            matrix[p][c] =1
    del matrix[nodeid]
    for row in matrix:
        del row[nodeid]

def constructAPIContextGraphEdges(codelines,apiContextGraphNodes,matrix):
    """
        api
        if
    """
    delNodes = []
    for i,row in enumerate(matrix):
        nodeLines = [ node['value'] for node in apiContextGraphNodes]

        if 0==i or codelines[i].strip() in nodeLines:
            pass
        else:
            delNodes.append(i)
    delNodes.reverse()
    for id in delNodes:
        delnode(id,matrix)


    for row,items in enumerate(matrix):
        apiContextGraphNodes[row].setdefault('children', [])
        for colum,num in enumerate(items):
            if 1 == num :
                apiContextGraphNodes[row]['children'].append(colum)
    return apiContextGraphNodes,[list(row) for row in matrix]



_REF = {javalang.tree.MemberReference,
        javalang.tree.ClassReference,
        javalang.tree.MethodInvocation}

_BLOCK = {'body',
          'block',
          'then_statement',
          'else_statement',
          'catches',
          'finally_block'}

_IGNORE = {'throws',
           'dimensions',
           'prefix_operators',
           'postfix_operators',
           'selectors',
           'types',
           'case'}

_LITERAL_NODE = {'Annotation',
                 'MethodDeclaration',
                 'ConstructorDeclaration',
                 'FormalParameter',
                 'ReferenceType',
                 'MemberReference',
                 'VariableDeclarator',
                 'MethodInvocation',
                 'Literal'}


def get_value(node, token_list):
    value = None
    length = len(token_list)
    if hasattr(node, 'name'):
        value = node.name
    elif hasattr(node, 'value'):
        value = node.value
    elif type(node) in _REF and node.position:
        for i, token in enumerate(token_list):
            if node.position == token.position:
                pos = i + 1
                value = str(token.value)
                while pos < length and token_list[pos].value == '.':
                    value = value + '.' + token_list[pos + 1].value
                    pos += 2
                break
    elif type(node) is javalang.tree.TypeArgument:
        value = str(node.pattern_type)
    elif type(node) is javalang.tree.SuperMethodInvocation \
            or type(node) is javalang.tree.SuperMemberReference:
        value = str(node.member)
    elif type(node) is javalang.tree.BinaryOperation:
        value = node.operator
    return value


def parse_single(code, max_nodes=max_nodes):
    # java lang
    tokens = javalang.tokenizer.tokenize(code)
    token_list = list(javalang.tokenizer.tokenize(code))
    parser = javalang.parser.Parser(tokens)
    try:
        tree = parser.parse_member_declaration()
    except:
        return []

    result = []
    q = deque([tree])
    idx = 1  # index of the next child node (level traversal)
    while len(q) > 0 and len(result) <= max_nodes:
        node = q.popleft()
        if type(node) is dict:
            result.append(node)
            continue
        node_d = {'id': len(result), 'type': node.__class__.__name__, 'children': []}
        value = get_value(node, token_list)
        if value is not None and type(value) is str:
            node_d['value'] = value
        result.append(node_d)

        for attr, child in zip(node.attrs, node.children):
            if idx >= max_nodes:
                break
            if attr in _BLOCK and child:
                if type(child) is javalang.tree.BlockStatement:
                    child = child.statements
                block_d = {'id': idx, 'type': attr, 'children': []}
                node_d['children'].append(idx)
                idx += 1
                q.append(block_d)
                node_d = block_d
            if isinstance(child, javalang.ast.Node):
                node_d['children'].append(idx)
                idx += 1
                q.append(child)
            elif type(child) is list and child and attr not in _IGNORE:
                child = [c[0] if type(c) is list else c for c in child[:max_nodes - idx]]
                child_idx = [idx + i for i in range(len(child))]
                node_d['children'].extend(child_idx)
                idx += len(child)
                q.extend(child)
    return result


def get_ast(codes, max_nodes=max_nodes, workers=workers, save_path=None):
    # java 代码的AST如何构造
    desc = 'Building ASTs...'
    if workers > 1 and len(codes) > 2000:
        #print(desc)
        func = partial(parse_single, max_nodes=max_nodes)  # 把某个函数的参数固定住，返回一个新的函数
        results = parallel(func, codes, workers=workers)
    else:
        results = []
        for code in tqdm(codes, desc=desc):  # 进度条？
            results.append(parse_single(code, max_nodes))
    #
    dropped = set(i for i, tree in enumerate(results) if len(tree) == 0)
    print('Number of parse failures:', len(dropped))  # AST 解析失败的节点  文法解析（LL（1））？
    if save_path is not None:
        save(results, save_path, is_json=True)
    return results, dropped


def node_filter(s, subtoken=subtoken):
    s = re.sub(r"\d+\.\d+\S*|0[box]\w*|\b\d+[lLfF]\b", " num ", s)
    s = re.sub(r"%\S*|[^A-Za-z0-9\s]", " ", s)
    s = re.sub(r"\b\d+\b", " num ", s)
    if subtoken:
        s = re.sub(r"[a-z][A-Z]", lambda x: x.group()[0] + " " + x.group()[1], s)
        s = re.sub(r"[A-Z]{2}[a-z]", lambda x: x.group()[0] + " " + x.group()[1:], s)
        s = re.sub(r"\w{32,}", " ", s)  # MD5, hash
        s = re.sub(r"[A-Za-z]\d+", lambda x: x.group()[0] + " ", s)
    s = re.sub(r"\s(num\s+){2,}", " num ", s)
    return s.lower().split()

def pre_traverse(tree, idx, node_maxlen, subtoken,results):
    node = tree[idx]
    if len(results)>=node_maxlen:
        return

    #results_type.append(node['type'])
    #if node['type'] in ['funcName','api_call']:
    #value = node_filter(node['value'], subtoken)
    #result=value[:node_maxlen]
    results.append(node['id'])
    #elif node.get('value'):
        #result.append(node['value'].lower())

    if node['children']:
        for child in node['children']:
           pre_traverse(tree, child, node_maxlen, subtoken,results)
    #return result


def get_node_seq(trees, node_maxlen=node_maxlen, subtoken=subtoken,
                 max_tokens=max_tokens, save_path=None):
    results = []
    for tree in tqdm(trees, desc='Obtaining node seqs...'):
        results.append(pre_traverse(tree, 0, node_maxlen, subtoken)[:max_tokens])
    if save_path is not None:
        save(results, save_path, is_json=True)
    return results


def split_ast(trees, node_maxlen=node_maxlen, max_statms=max_statms,
              min_statms=min_statms, subtoken=subtoken, save_path=None):
    def traverse(tree, idx):
        node = tree[idx]
        subTrees = []
        blocks = []
        for i, child in enumerate(node['children']):
            if tree[child]['type'] in _BLOCK:
                blocks.append(child)
                del node['children'][i]
        subTrees.append(pre_traverse(tree, idx, node_maxlen, subtoken)[:max_tokens])
        for block in blocks:
            for child in tree[block]['children']:
                subTrees.extend(traverse(tree, child))
        return subTrees

    results = []
    dropped = set()
    for idx, tree in enumerate(tqdm(trees, desc='Splitting ASTs...')):
        result = traverse(tree, 0)[:max_statms]
        results.append(result)
        if len(result) < min_statms:
            #
            dropped.add(idx)
    if save_path is not None:
        save(results, save_path, is_json=True)
    return results, dropped

def extractAPI(tree):
    apiList = []
    for node in tree:
        if 'MethodInvocation' == node['type']:
            apiList.append(node['value'])
    return apiList

def split_node(node,nodeid,apiGraph,matrix):
    if 'Method_CALL' == node['type'] or 'Condition' == node['type']:
        newNodeLabel = node['api'].pop(0)
        newNode = {}
        newNode['id'] = len(apiGraph)+1
        newNode['value'] = node['value']
        newNode['type'] = node['type']
        newNode['api'] = [newNodeLabel]
        newNode['label'] =newNodeLabel
        apiGraph.insert(nodeid,newNode)

        matrix.insert(nodeid,len(matrix)*[0])
        for i in range(len(matrix)):
            matrix[i].insert(nodeid,matrix[i][nodeid])
            matrix[i][nodeid+1]=0
        matrix[nodeid][nodeid+1] =1


def get_graph_seqs(code,api_seqs):
    import pyastyle as style
    import re
    #code = re.sub('/\*.*?\*/', '', code).strip()
    code = style.format(code, '--style=allman --delete-empty-lines --add-brackets')
    code = re.sub('//.*?\n', '', code).strip()
    #print(code)
    codelines = code.split('\n')
    # 从第三行开始
    Nodes = constructAPIContextGraphNodes(codelines, api_seqs)
    matrix = parseData(codelines)
    apiGraph, matrix = constructAPIContextGraphEdges(codelines, Nodes, matrix)
    # 分裂节点；
    for i,node in enumerate(apiGraph):
        if 'Method_CALL'== node['type'] or 'Func_Name'== node['type'] or 'Condition'== node['type']:
            while 1 !=len(node['api']):
                split_node(node,apiGraph.index(node),apiGraph,matrix)
            node['label'] = node['api'][0]  # 不用更改
        else:  # 控制类节点 加 condition 节点
            #split_node(node,i,apiGraph,matrix)
            if None != node['api']:
                node['label'] = node['type']
                newNode = {}
                newNode['id'] = len(apiGraph) + 1
                newNode['value'] = node['value']
                newNode['type'] = 'Condition'
                newNode['api'] = node['api']
                #newNode['label'] = node['type']
                apiGraph.insert(apiGraph.index(node) + 1,newNode)

                nodeid = apiGraph.index(node)
                t= matrix[nodeid]
                matrix.insert(nodeid + 1,t)
                matrix[nodeid] = [0]*len(matrix)
                for i in range(len(matrix)):
                    matrix[i].insert(nodeid+1,0)
                matrix[nodeid][nodeid+1] = 1
                # while 1 != len(newNode['api']):
                #      split_node(newNode, apiGraph.index(newNode), apiGraph, matrix)

            else:
                node['label'] = node['type']



    results_nodeseqs = []
    results_types = []

    import numpy as np
    import config
    result_matrix = np.zeros((config.max_node_length + 1, config.max_node_length + 1), dtype=np.int32)



    maxtrix_len = min(len(apiGraph),config.max_node_length)
    for (i,node) in enumerate(apiGraph):
        #value = node_filter(apiGraph[nodeId]['value'], subtoken)
        if i>=config.max_node_length:
            break

        #value = node['label']
        value =node_filter(node['label'], subtoken)
        results_nodeseqs.append(value)
        results_types.append(node['type'])
        try:
            for j in range(maxtrix_len):
                result_matrix[i][j] = matrix[i][j]
        except:
            print('result_matrix',result_matrix)
            print('matrix',matrix)
    return results_nodeseqs,results_types,result_matrix





