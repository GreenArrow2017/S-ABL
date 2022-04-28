# from utils import *

import queue as Q
import numpy as np
import math
import time
import signal
import re
# sym2priority = {'+': 0, '-': 0, '*': 1, '/': 1}
# sym2priority.update({str(x):2 for x in digit_list})

# NAN_THRESHOLD = 10e7
# thres_nan = lambda x: x if abs(eval(x)) < NAN_THRESHOLD else float('nan')
# plus = lambda x,y: thres_nan(eval(x) + eval(y))
# minus = lambda x,y: thres_nan(eval(x) - eval(y))
# times = lambda x,y: thres_nan(eval(x) * eval(y))
# divide = lambda x,y: thres_nan(eval(x) / eval(y) if eval(y) != 0 else float('nan'))
# exp = lambda x,y: thres_nan(eval(x) ** eval(y) if abs(eval(x)) < 10000 and eval(y) <1000 else float('nan'))
# root = lambda x,y: thres_nan(exp(eval(x), divide(1, eval(y))))
# log = lambda x,base: thres_nan(math.log(eval(x), base) if base != 0 and base != 1 and eval(x) > 0 else float('nan'))
# symbol2semantic= {'+': plus, '-': minus, '*': times, '/': divide, '^': exp}
# #symbol2semantic.update({x: eval(x) if x.isdigit()})
# inverse_op_left = {'+': minus, '-': plus, '*': divide, '/': times, '^': root}
# inverse_op_right = {
#     '+': minus,
#     '-': lambda target, left: minus(left, target),
#     '*': divide,
#     '/': lambda target, left: divide(left, target),
#     '^': log}

NAN_THRESHOLD = 10e7
thres_nan = lambda x: x if abs(x) < NAN_THRESHOLD else float('nan')
plus = lambda x, y: thres_nan(x + y)
minus = lambda x, y: thres_nan(x - y)
times = lambda x, y: thres_nan(x * y)
divide = lambda x, y: thres_nan(x / y if y != 0 and y != 1 else float('nan'))
exp = lambda x, y: thres_nan(x**y if abs(x) < 1000 and abs(y) < 10 and x != 1 and y != 1 else float('nan'))
root = lambda x, y: thres_nan(exp(x, divide(1, y)))
log = lambda x, base: thres_nan(math.log(x, base) if base > 0 and base != 1 and x > 0 else float('nan'))
# NAN_THRESHOLD = 10e7
# thres_nan = lambda x: x if abs(x) < NAN_THRESHOLD else 1e5
# plus = lambda x,y: thres_nan(x + y)
# minus = lambda x,y: thres_nan(x - y)
# times = lambda x,y: thres_nan(x * y)
# divide = lambda x,y: thres_nan(x / y if y != 0 else 1e5)
# exp = lambda x,y: thres_nan(x ** y if abs(x) < 1000 and y < 10 and x != 1 and y != 1 else 1e5)
# root = lambda x,y: thres_nan(exp(x, divide(1, y)))
# log = lambda x,base: thres_nan(math.log(x, base) if base > 0 and base != 1 and x > 0 else 1e5)
symbol2semantic = {'+': plus, '-': minus, '*': times, '/': divide, '^': exp, '**': exp}
inverse_op_left = {'+': minus, '-': plus, '*': divide, '/': times, '^': root, '**': root}
inverse_op_right = {'+': minus, '-': lambda target, left: minus(left, target), '*': divide, '/': lambda target, left: divide(left, target), '^': log, '**': log}


class LeafNode:
    def __init__(self, symbol, all_prob, sym_list, num_start):
        self.symbol = symbol
        self.all_prob = all_prob - np.log(np.sum(np.exp(all_prob)))
        self.sym_list = sym_list
        self.num_start = num_start
        self.initialize()

    def initialize(self):
        self.symbol_id = self.sym_list.index(self.symbol)
        self.prob = self.all_prob[self.symbol_id]
        self.max_prob = self.all_prob.max()
        self.parent = None
        if self.symbol in symbol2semantic:
            self._res = symbol2semantic[self.symbol]
        else:
            self._res = self.symbol

    def res(self):
        return [self._res, self.prob, self.max_prob]

    def entropy(self):
        return -1 * np.sum(np.exp(self.all_prob) * self.all_prob)

    def sample(self):
        # self.all_prob[self.symbol_id] = np.log(1e-30)
        # self.all_prob = self.all_prob - np.log(np.sum(np.exp(self.all_prob)))

        all_prob = np.exp(self.all_prob)
        all_prob_new = all_prob
        if self.symbol in symbol2semantic:
            all_prob_new[self.num_start:] = 0
        else:
            all_prob_new[:self.num_start] = 0
        all_prob_new[self.sym_list.index(self.symbol)] = 1e-6
        all_prob_new /= all_prob_new.sum()
        new_symbol = np.random.choice(self.sym_list, p=all_prob_new)

        if isinstance(new_symbol, str) and any(char.isdigit() for char in new_symbol):
            new_symbol = float(new_symbol)

        self.prev_symbol = self.symbol
        self.symbol = new_symbol

        self.initialize()
        return self.symbol

    def resume(self):
        self.symbol = self.prev_symbol
        self.initialize()


class Node:
    def __init__(self, left, right, op):
        self.left = left
        self.right = right
        self.op = op
        self.parent = None
        self._res = None  # (res, prob, max_prob)
        self.prob = None
        self.max_prob = None

    def res(self):
        if self._res != None:
            return self._res
        left_res = self.left.res()
        right_res = self.right.res()

        op_res = self.op.res()
        prob = left_res[1] + right_res[1] + op_res[1]
        max_prob = left_res[2] + right_res[2] + op_res[2]
        try:
            res = op_res[0](left_res[0], right_res[0])
        except:
            res = float('nan')
        self._res = [res, prob, max_prob]
        self.prob = prob
        self.max_prob = max_prob
        return self._res


from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


class ExprTree:
    def __init__(self, sym_list, num_start):
        self.tokens = None
        self.root = None
        self.sym_list = sym_list
        self.num_start = num_start

    def handeler(self, signo, frame):
        print("runtime error")
        raise RuntimeError

    def parse(self, tokens=None):
        if tokens is not None:
            tokens = [LeafNode(*tok, self.sym_list, self.num_start) for tok in tokens]
            self.tokens = tokens
        else:
            tokens = self.tokens

        values = []
        operators = []

        for token in reversed(tokens):
            if token.symbol not in ["+", "-", "*", "/", "^", "**"]:
                values.append(token)
            else:
                op = token
                left = values.pop()
                right = values.pop()
                new_node = Node(left, right, op)
                op.parent = new_node
                right.parent = new_node
                left.parent = new_node
                values.append(new_node)

        self.root = values.pop()
        self.root.res()
        return self.root

    def res(self):
        return self.root.res()

    def get_subtree_target(self, node, target):
        number_list = self.sym_list[self.num_start:]
        operator = self.operator
        node_id = self.tokens.index(node)
        for index1, parameter1 in enumerate(number_list):
            for index2, parameter2 in enumerate(number_list):
                if index1 != index2:
                    for op in operator:
                        if (abs(symbol2semantic[op](parameter1, parameter2) - target) <= 1e-5):
                            # a/a = 1  1^n = 1  n^1 = n
                            if (parameter1 == parameter2 and op == '/') and (op == '**' and parameter1 == 1) and (op == '**' and parameter2 == 1) and (op == '*' and parameter1 == 1) and (
                                    op == '*' and parameter2) and (op == '-' and parameter1 == parameter2) and (op == '/' and parameter2 == 1):
                                continue
                            leftNode = LeafNode(parameter1, self.all_prob[node_id], self.sym_list, self.num_start)
                            rightNode = LeafNode(parameter2, self.all_prob[node_id], self.sym_list, self.num_start)
                            opera = LeafNode(op, self.all_prob[node_id], self.sym_list, self.num_start)
                            new_node = Node(leftNode, rightNode, opera)
                            leftNode.parent = new_node
                            rightNode.parent = new_node
                            opera.parent = new_node
                            new_node.res()
                            return new_node
        return None

    def find_valid_change_dynamic(self, node, target, op):
        if isinstance(node, LeafNode):
            find = False
            for sym in self.sym_list:
                if not isinstance(sym, str):
                    if not (op == "**" and sym == 1):
                        if abs(target - sym) < 1e-5:
                            change = PrioritizedItem(node.prob - node.all_prob[self.sym_list.index(sym)], (node, target, sym))
                            find = True
            if not find:
                # 增加新的分支
                new_node = self.get_subtree_target(node, target)
                if new_node != None:
                    change = PrioritizedItem(node.prob - node.max_prob, (node, target, new_node))
                    find = True
                else:
                    change = None
        else:
            change = PrioritizedItem(node.prob - node.max_prob, (node, target))
        return change

    def compute_prefix_expression(self, pre_fix):
        st = list()
        operators = ["+", "-", "**", "*", "/"]
        pre_fix.reverse()
        for p in pre_fix:
            if p not in operators:
                pos = re.search("\d+\(", p)
                if pos:
                    st.append(eval(p[pos.start():pos.end() - 1] + "+" + p[pos.end() - 1:]))
                elif p[-1] == "%":
                    st.append(float(p[:-1]) / 100)
                else:
                    st.append(eval(p))
            elif p == "+" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(a + b)
            elif p == "*" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(a * b)
            elif p == "/" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                if b == 0 or b == 1:
                    return None
                st.append(a / b)
            elif p == "-" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(a - b)
            elif p == "**" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                if float(b) != 2.0 or float(b) != 3.0:
                    return None
                st.append(a**b)
            else:
                return None
        if len(st) == 1:
            return st.pop()
        return None

    def fix_1step_dynamic(self, gt):
        olds = [tok.symbol for tok in self.tokens]

        queue = Q.PriorityQueue()
        change = PrioritizedItem(0., (self.root, gt))
        queue.put(change)

        while not queue.empty():
            change = queue.get()
            prob = change.priority
            node, target, *rest = change.item
            if isinstance(node, LeafNode):
                token_idx = self.tokens.index(node)
                if len(change.item) >= 3:
                    if isinstance(change.item[2], Node):
                        print("找到分支，改变树的结构")
                        # 说明是经过了节点扩展
                        new_node = change.item[2]
                        new_token = [new_node.op.symbol, new_node.left.symbol, new_node.right.symbol]
                        news = olds.copy()
                        news[token_idx] = new_token
                        news_return = []
                        for tokens in news:
                            if isinstance(tokens, list):
                                for token in tokens:
                                    news_return.append(token)
                            else:
                                news_return.append(tokens)
                        # print(news_return, new_token, gt)
                        return (news_return, self.root.res()[1] - prob)
                    # 否则就是支点替换
                    target_sym = change.item[2]
                    # olds不是tokens，是一个symbol的list
                    news = olds.copy()
                    news[token_idx] = target_sym
                    # 直接结束
                    # 所以一个fix_1step()只会找到一个fix equation
                    return (news, self.root.res()[1] - prob)
                else:
                    return None

            left = node.left
            right = node.right
            op = node.op

            if right.res()[0] == float('nan') or left.res()[0] == float('nan'):
                return None
            # change left
            try:
                sub_target = inverse_op_left[op.symbol](target, right.res()[0])
                if sub_target == float('nan'):
                    change = None
                else:
                    change = self.find_valid_change_dynamic(left, sub_target, op.symbol)
            except:
                change = None
            if change is not None:
                queue.put(change)

            # change right
            try:
                sub_target = inverse_op_right[op.symbol](target, left.res()[0])
                if sub_target == float('nan'):
                    change = None
                else:
                    change = self.find_valid_change_dynamic(right, sub_target, op.symbol)
            except:
                change = None
            if change is not None:
                queue.put(change)

            # change op
            ori_op = op.symbol
            token_idx = self.tokens.index(op)
            sub_target = None

            for new_op in symbol2semantic.keys():
                if new_op == ori_op:
                    continue

                new_exp = [tok.symbol for tok in self.tokens]
                new_exp[token_idx] = "**" if new_op == "^" else new_op
                for j in range(len(new_exp)):
                    if not isinstance(new_exp[j], str):
                        new_exp[j] = str(new_exp[j])

                new_res = self.compute_prefix_expression(new_exp)
                #print (new_res)
                if not new_res:
                    continue
                if abs(new_res - gt) < 1e-5:
                    sub_target = new_op
                    change = PrioritizedItem(op.prob - op.all_prob[self.sym_list.index(sub_target)], (op, sub_target, sub_target))
                    queue.put(change)

        return None
    
    def fix_1step(self, gt):
        olds = [tok.symbol for tok in self.tokens]

        queue = Q.PriorityQueue()
        change = PrioritizedItem(0., (self.root, gt))
        queue.put(change)

        while not queue.empty():
            change = queue.get()
            prob = change.priority
            node, target, *rest = change.item
            if isinstance(node, LeafNode):
                # print('find a fix, early stop.')
                token_idx = self.tokens.index(node)

                if len(change.item) >= 3: # if target_sym exists
                    target_sym = change.item[2]
                    news = olds.copy()
                    news[token_idx] = target_sym
                    return (news, self.root.res()[1] - prob)
                else:
                    return None

            left = node.left
            right = node.right
            op = node.op

            if right.res()[0] == float('nan') or left.res()[0] == float('nan'):
                return None
            # change left
            try:
                sub_target = inverse_op_left[op.symbol](target, right.res()[0])
                if sub_target == float('nan'):
                    change = None
                else:
                    change = self.find_valid_change(left, sub_target, op.symbol)
            except:
                change = None
            if change is not None:
                queue.put(change)

            # change right
            try:
                sub_target = inverse_op_right[op.symbol](target, left.res()[0])
                if sub_target == float('nan'):
                    change = None
                else:
                    change = self.find_valid_change(right, sub_target, op.symbol)
            except:
                change = None
            if change is not None:
                queue.put(change)

            # change op
            ori_op = op.symbol
            token_idx = self.tokens.index(op)
            sub_target = None

            for new_op in symbol2semantic.keys():
                if new_op == ori_op:
                    continue

                new_exp = [tok.symbol for tok in self.tokens]
                new_exp[token_idx] = "**" if new_op=="^" else new_op
                for j in range(len(new_exp)):
                    if not isinstance (new_exp[j], str):
                        new_exp[j] = str(new_exp[j])

                new_res = self.compute_prefix_expression(new_exp)
                #print (new_res)
                if not new_res:
                    continue
                if abs(new_res - gt) < 1e-12:
                    sub_target = new_op
                    change = PrioritizedItem(op.prob - op.all_prob[self.sym_list.index(sub_target)], (op, sub_target, sub_target))

                    queue.put(change)
        return None

    def find_valid_change(self, node, target, op):
        if isinstance(node, LeafNode):
            find = False
            for sym in self.sym_list:
                if not isinstance (sym, str):
                    if not (op == "**" and sym == 1):
                        if abs(target - sym) < 1e-12:
                            change = PrioritizedItem(node.prob - node.all_prob[self.sym_list.index(sym)], (node, target, sym))
                            find = True
            if not find:
                change = None
        else:
            change = PrioritizedItem(node.prob - node.max_prob, (node, target))
        return change
    def fix(self, gt, n_step=1):
        entropy_list = np.array([x.entropy() for x in self.tokens])
        entropy_list = entropy_list / entropy_list.sum()
        res_list = []

        for i in range(n_step):
            if i > 0:
                self.parse()
                # results = [tok.symbol for tok in self.tokens]
                # # res = [tok._res for tok in self.tokens]
                # print (results)
                # # print (res)
                # print (self.res())

            fix = self.fix_1step(gt)

            if fix is not None:
                return fix
            else:
                accept = False
                not_accept_times = 0
                while not accept and not_accept_times <= 5:
                    not_accept_times += 1
                    n_sym_change = int(np.abs(np.random.normal(0, 1, 1)))
                    n_sym_change = np.maximum(n_sym_change, 1)
                    n_sym_change = np.minimum(n_sym_change, len(self.tokens))

                    prob_old_string = np.sum([x.prob for x in self.tokens])
                    token_ids = np.random.choice(len(self.tokens), n_sym_change, replace=False)
                    results = [tok.symbol for tok in self.tokens]
                    for tok_id in token_ids:
                        self.tokens[tok_id].sample()
                    prob_new_string = np.sum([x.prob for x in self.tokens])
                    accept_ratio = np.exp(prob_new_string - prob_old_string)
                    if np.random.random() < accept_ratio:
                        results = [tok.symbol for tok in self.tokens]
                        if results not in res_list:
                            res_list.append(results)
                            accept = True
                        else:
                            accept = False
                            for tok_id in token_ids:
                                self.tokens[tok_id].resume()
                    else:
                        for tok_id in token_ids:
                            self.tokens[tok_id].resume()

        return None

    def fix_bak(self, gt, n_step=1):
        entropy_list = np.array([x.entropy() for x in self.tokens])
        entropy_list = entropy_list / entropy_list.sum()
        print([x.symbol for x in self.tokens])
        for i in range(n_step):
            if i > 0:
                self.parse()
            fix = self.fix_1step(gt)
            if fix is not None:
                return fix
            else:
                token_id = np.random.choice(entropy_list.shape[0], p=entropy_list)
                new_symbol = self.tokens[token_id].sample()
                print([x.symbol for x in self.tokens])
        return None



