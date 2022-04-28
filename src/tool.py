import re
from typing import runtime_checkable
from numpy.core.numeric import False_
import torch
import numpy as np
import random
from src.expressions_transfer import compute_prefix_expression, out_expression_list
from copy import deepcopy

train_fix_templates = {}

class node:
    def __init__(self, index, symbol, isleaf=True, lchild=None, rchild=None):
        self.index = index
        self.symbol = symbol
        self.isleaf = isleaf
        self.lchild = lchild
        self.rchild = rchild

    def get_symbol(self):
        symbol = deepcopy(self.symbol)
        return symbol

    def get_index(self):
        index = deepcopy(self.index)
        return index

    def leaf(self):
        return self.isleaf


def build_tree(post_index, post_exp, num_start):
    iscorrect = True
    stack = []
    for i, e in zip(post_index, post_exp):
        if i >= num_start:
            # 说明是数字
            leaf_ = node(i, e, isleaf=True)
            stack.append(leaf_)
        else:
            # 说明是操作符
            if len(stack) < 2:
                # 数字数量不够，不能构成一棵树，不合法，返回
                iscorrect = False
                return None, iscorrect
            lchild = stack.pop()
            rchild = stack.pop()
            node_ = node(i, e, isleaf=False, lchild=lchild, rchild=rchild)
            stack.append(node_)
    root = stack[-1]
    return root, iscorrect


def simplify_tree(root, num_start):
    '''
    +: A+0, 0+A
    -: A-0, A-A
    /: A/1, A/A, 0/A, A/0, 1/(A/B)
    *: A*1, 1*A, A*0, 0*A
    ^: A^1, A^0, 1^A, 0^A
    包含以上结构的等式需要化简
    '''
    zero_index = 10000
    zero = '0'
    one_index = 5
    one = '1'
    ilegal_index = -1
    ilegal = '-1'
    if root == None or root.leaf():
        return root

    if root.get_symbol() == '+':
        root.lchild = simplify_tree(root.lchild, num_start)
        root.rchild = simplify_tree(root.rchild, num_start)
        if root.lchild.get_symbol() == '0':
            return root.rchild
        if root.rchild.get_symbol() == '0':
            return root.lchild

    if root.get_symbol() == '-':
        root.lchild = simplify_tree(root.lchild, num_start)
        root.rchild = simplify_tree(root.rchild, num_start)
        if root.rchild.get_symbol() == '0':
            return root.lchild
        if root.lchild.leaf() and root.rchild.leaf():
            # 如果都是数字
            if root.lchild.get_symbol() == root.rchild.get_symbol():
                node_ = node(zero_index, zero, isleaf=True)
                return node_

    if root.get_symbol() == '*':
        root.lchild = simplify_tree(root.lchild, num_start)
        root.rchild = simplify_tree(root.rchild, num_start)
        if root.lchild.get_symbol() == '1':
            return root.rchild
        if root.rchild.get_symbol() == '1':
            return root.lchild
        if root.lchild.get_symbol() == '0' or root.rchild.get_symbol() == '0':
            node_ = node(zero_index, zero, isleaf=True)
            return node_

    if root.get_symbol() == '/':
        root.lchild = simplify_tree(root.lchild, num_start)
        root.rchild = simplify_tree(root.rchild, num_start)
        if root.rchild.get_symbol() == '0':
            node_ = node(ilegal_index, ilegal, isleaf=True)
            return node_
        if root.lchild.get_symbol() == '0':
            node_ = node(zero_index, zero, isleaf=True)
            return node_
        if root.rchild.get_symbol() == '1':
            return root.lchild
        if root.lchild.leaf() and root.rchild.leaf():
            if root.lchild.get_symbol() == root.rchild.get_symbol():
                node_ = node(one_index, one, isleaf=True)
                return node_
        if root.lchild.get_symbol() == '1' and root.rchild.get_symbol() == '/':
            temp = deepcopy(root.rchild.lchild)
            root.rchild.lchild = root.rchild.rchild
            root.rchild.rchild = temp
            return root.rchild

    if root.get_symbol() == '^' or root.get_symbol() == '**':
        root.lchild = simplify_tree(root.lchild, num_start)
        root.rchild = simplify_tree(root.rchild, num_start)
        if root.rchild.get_symbol() == '1':
            return root.lchild
        if root.rchild.get_symbol() == '0':
            return node(one_index, one, isleaf=True)
        if root.lchild.get_symbol() == '1':
            return node(one_index, one, isleaf=True)
        if root.lchild.get_symbol() == '0':
            return node(zero_index, zero, isleaf=True)
    return root


def tree2fix(node, fix_list):
    if node == None:
        return
    fix_list.append(node.get_index())
    tree2fix(node.lchild, fix_list)
    tree2fix(node.rchild, fix_list)


def simplify_equation(fix, fix_exp, num_start, output_lang, numList):
    post_index = deepcopy(fix)
    post_exp = deepcopy(fix_exp)
    post_index.reverse()
    post_exp.reverse()
    root, iscorrect = build_tree(post_index, post_exp, num_start)
    if iscorrect == False:
        return None, False
    root = simplify_tree(root, num_start)
    new_fix = []
    islegal = True
    tree2fix(root, new_fix)
    zero = False
    for i in new_fix:
        if i == -1:
            islegal = False
        if i == 10000:
            zero = True
    if zero:
        new_fix = fix
    if islegal == False:
        return new_fix, islegal
    else:
        return new_fix, ilegal_subtitute(fix, output_lang, numList)


def is_AABB(fix, wordindex):
    format_eq = [wordindex.index('/'), wordindex.index('-'), 'N', 'N', wordindex.index('-'), 'Y', 'Y']
    indexs = [0, 1, 4]
    if len(fix) != len(format_eq):
        return False
    for index in indexs:
        if fix[index] != format_eq[index]:
            return False
    if fix[2] == fix[3] and fix[5] == fix[6]:
        return True


def ilegal(fix, wordindex):
    return is_AABB(fix, wordindex)


def ilegal_subtitute(fix, output_lang, numList):
    cp_fix = deepcopy(fix)
    num_sub = []
    for num in numList:
        num_sub.append(str(random.randint(50000, 60000)))
    result1 = compute_prefix_expression(out_expression_list(cp_fix, output_lang, numList))
    result2 = compute_prefix_expression(out_expression_list(cp_fix, output_lang, num_sub))
    if result1 == result2:
        return False
    return True


def usage(equation, num_list):
    count = 0
    for num in num_list:
        if num in equation:
            count += 1
    usage_ = count / len(num_list)

    return usage_


class TreeNode:
    def __init__(self, left, right, sym, parent, isleaf):
        self.left = left
        self.right = right
        self.sym = sym
        self.parent = parent
        self.isleaf = isleaf


class math_property_transformer:
    # 对等式生成交换, 分配
    # 输入一条等式，输出交换分配的等式。
    def __init__(self, sym_list, equation):
        # num_list是出现在文本的数字
        # sym_list是包含操作符所有的数字
        self.equation = equation.copy()
        self.sym_list = sym_list
        self.operaters = ["+", "-", "*", "/", "**", "^"]
        self.priority = {"+": 1, "-": 1, "*": 2}
        self.commutative = {"+", "*"}
        self.associative = ['+', '-']
        self.root = self.equation2tree()

    def generate_commutative(self):
        # 交换律
        equation_commutative = []
        root = deepcopy(self.root)
        opers = list()
        if root.isleaf == False:
            # 说明是操作节点
            opers.append(root)
        while len(opers) > 0:
            node = opers.pop()
            if node.sym in self.commutative:
                # * +可以交换律
                left = node.left
                node.left = node.right
                node.right = left

                equation_commutative.append(self.tree2equation(node))

                # 换回来
                left = node.left
                node.left = node.right
                node.right = left

            if node.left.isleaf == False:
                # 左边也是操作符
                opers.append(node.left)
            if node.right.isleaf == False:
                # 右边也是操作符
                opers.append(node.right)
        return equation_commutative

    def generate_distributive(self):
        # 分配律 只有乘法
        root = deepcopy(self.root)
        opers = list()
        if root.isleaf == False:
            opers.append((root, None))
        while len(opers) > 0:
            node, position = opers.pop()

            if node.sym in self.priority.keys():
                # 幂次是不能做分配率的
                if node.left.sym in self.priority.keys() and (self.priority[node.sym] > self.priority[node.left.sym]):
                    # 左边，可以做分配率
                    parent = node.parent
                    left = node.left
                    right = node.right
                    sub_left = left.left
                    sub_right = left.right

                    left.left = TreeNode(sub_left, right, node.sym, left, False)
                    left.right = TreeNode(sub_right, right, node.sym, left, False)
                    left.parent = parent

                    if position != None:
                        # 说明是有爸爸的
                        if position == "left":
                            parent.left = left
                        if position == "right":
                            parent.right = left

                    node = left
                    return self.tree2equation(node)
                if node.right.sym in self.priority.keys() and (self.priority[node.sym] > self.priority[node.right.sym]):
                    # 右边，可以做分配率
                    parent = node.parent

                    right = node.right
                    left = node.left
                    sub_left = right.left
                    sub_right = right.right

                    right.left = TreeNode(left, sub_left, node.sym, right, False)
                    right.right = TreeNode(left, sub_right, node.sym, right, False)
                    right.parent = parent

                    if position != None:
                        # 说明是有爸爸的
                        if position == "left":
                            parent.left = right
                        if position == "right":
                            parent.right = right

                    node = right
                    return self.tree2equation(node)
            if node.sym == '/' and (node.left.sym == '+' or node.left.sym == '-'):
                # 除法分配
                parent = node.parent

                right = node.right
                left = node.left
                sub_left = left.left
                sub_right = left.right

                left.left = TreeNode(sub_left, right, node.sym, left, False)
                left.right = TreeNode(sub_right, right, node.sym, left, False)
                left.parent = parent

                if position != None:
                    if position == 'left':
                        parent.left = left
                    if position == 'right':
                        parent.right = left

                node = left
                print('除法分配')
                return self.tree2equation(node)
            else:
                if node.left.isleaf == False:
                    opers.append((node.left, "left"))
                if node.right.isleaf == False:
                    opers.append((node.right, "right"))

    def generate_associative(self):
        # 结合律
        root = deepcopy(self.root)
        opers = list()
        if root.isleaf == False:
            opers.append((root, None))
        while len(opers) > 0:
            node, position = opers.pop()

            if node.sym in self.associative:
                # 乘法结合律
                left = node.left
                right = node.right
                parent = node.parent

                if left.isleaf == False and right.isleaf == False and (left.sym == '*' and right.sym == '*'):
                    # 子节点不是数字，如果是乘法就可以进行结合了
                    left_child = [left.left, left.right]
                    right_child = [right.left, right.right]

                    lindex = -1
                    rindex = -1
                    for li, l in enumerate(left_child):
                        for ri, r in enumerate(right_child):
                            if (l.sym == r.sym) and (l.isleaf and r.isleaf):
                                # 找到了相同的节点
                                lindex = li
                                rindex = ri
                                break
                    if lindex != -1:
                        associate_node = deepcopy(left_child[lindex])
                        left_child.pop(lindex)
                        right_child.pop(rindex)
                        new_root = TreeNode(associate_node, None, left.sym, parent, False)
                        new_right = TreeNode(left_child[0], right_child[0], node.sym, new_root, False)
                        new_root.right = new_right

                        if position != None:
                            if position == 'left':
                                parent.left = new_root
                            if position == 'right':
                                parent.right = new_root
                        node = new_root
                        print("乘法结合")
                        return self.tree2equation(node)
                if left.isleaf == False and right.isleaf == False and (left.sym == '/' and right.sym == '/'):
                    # 除法结合律只有单向
                    if (left.right.isleaf and right.right.isleaf) and (left.right.sym == right.right.sym):
                        sub_left = left.left
                        sub_right = right.left
                        sub_root = TreeNode(sub_left, sub_right, node.sym, None, False)
                        new_root = TreeNode(sub_root, left.right, left.sym, parent, False)
                        sub_root.parent = new_root

                        if position != None:
                            if position == 'left':
                                parent.left = new_root
                            if position == 'right':
                                parent.right = new_root
                        node = new_root
                        print("除法结合")
                        return self.tree2equation(node)
            else:
                if node.left.isleaf == False:
                    opers.append((node.left, "left"))
                if node.right.isleaf == False:
                    opers.append((node.right, "right"))

    def equation2tree(self):
        # 生成一棵树
        equation_exp = self.equation
        stack = list()
        for e in reversed(equation_exp):
            if e not in self.operaters:
                # 如果是数字
                e = e.replace("%", '/100')
                stack.append(TreeNode(None, None, eval(e), None, True))
            else:
                left = stack.pop()
                right = stack.pop()
                node = TreeNode(left, right, e, None, False)
                left.parent = node
                right.parent = node
                stack.append(node)
        if len(stack) == 1:
            return stack[-1]
        else:
            print("equation出现异常，构建expression_tree出现异常......")
            print(self.equation)
            print(self.equation_exp)
            return None

    def tree2equation(self, node):
        # 这个node不一定是根节点，要通过node找到根节点再转equation
        # 树转等式
        equation_generate = []

        while node.parent != None:
            # 找爸爸先，什么时候没爸爸什么时候就是根节点
            node = node.parent
        stack = list()
        stack.append(node)
        while len(stack) > 0:
            current = stack.pop()
            equation_generate.append(current.sym)
            if current.isleaf == False:
                stack.append(current.right)
                stack.append(current.left)
        return [self.sym_list.index(i) for i in equation_generate]


def add_templates(templates, equation):
    expression = equation.copy()
    operaters = ['*', '/', '^', '**', '+', '-']
    num_count = 0
    for i in range(len(expression)):
        # 数字替换成NUM，操作符替换成op
        if expression[i] in operaters:
            expression[i] = 'OP'
        else:
            expression[i] = 'NUM'
            num_count += 1
    if num_count not in templates.keys():
        templates[num_count] = []
        templates[num_count].append(expression)
    else:
        if expression not in templates[num_count]:
            templates[num_count].append(expression)
    return templates

def choose_equation_from_templates(templates, sym_list, num_start, operator):
    equations = []
    sys_list = sym_list
    number_list = sys_list[num_start:]
    # 问题含有数字的数量
    num_count = len(number_list)
    # 有的问题找不到solution的原因是文本出现了没用的数字
    # 模板数字的数量
    nums = [num_count, num_count - 1, num_count - 2]
    for num in nums:
        # 根据数字选择模板
        if num in templates.keys():
            template = templates[num]
            # 遍历模板
            for t in template:
                temp = t.copy()
                if num_count == num:
                    index = 0
                    for i in range(len(temp)):
                        if temp[i] == 'NUM':
                            temp[i] = number_list[index]
                            index += 1
                if num_count > num:
                    index_c = random.sample(range(0, num_count), num)
                    index = 0
                    for i in range(len(temp)):
                        if temp[i] == 'NUM':
                            temp[i] = number_list[index_c[index]]
                            index += 1
                if num_count < num:
                    index_c = random.sample(range(0, num), num_count)
                    index = 0
                    num_index = 0
                    for i in range(len(t)):
                        if temp[i] == 'NUM':
                            if index in index_c:
                                temp[i] = number_list[num_index]
                                num_index += 1
                            else:
                                temp[i] = (sym_list[num_start:])[random.randint(0, len(sym_list[num_start:]) - 1)]
                            index += 1
                for i in range(len(t)):
                    if temp[i] == 'OP':
                        temp[i] = operator[random.randint(0, len(operator) - 1)]
                equations.append(temp)
    return equations
