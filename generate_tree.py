from __future__ import print_function
import argparse
import os
import shutil
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dir', help="data directory", default="data-orchard")
parser.add_argument('--max_args', help="max number of integers per value node", type=int, default=2)
parser.add_argument('--max_depth', help="max depth of tree", type=int, default=6)
parser.add_argument('--size', help="size of data set in 000s", type=int, default=10)
parser.add_argument('--mm', help='add min max', action='store_true', default=False)
parser.add_argument('--fl', help='add first last', action='store_true', default=False)
parser.add_argument('--ms', help='add med summod', action='store_true', default=False)
parser.add_argument('--all', help='use all', action='store_true', default=False)
parser.add_argument('--p', help="probability of branching tree", type=float, default=0.5)
parser.add_argument('--c', help="probabilty of generating copy operator", type=float, default=0.5)
args = parser.parse_args()

COPY = "[COPY"
FIRST = "[FIRST"
LAST = "[LAST"
MIN = "[MIN"
MAX = "[MAX"
MED = "[MED"
SUM_MOD = "[SM"
END = "]"

OPERATORS = []
if args.mm:
	OPERATORS.append(MIN)
	OPERATORS.append(MAX)
if args.fl:
	OPERATORS.append(FIRST)
	OPERATORS.append(LAST)
if args.ms:
	OPERATORS.append(MED)
	OPERATORS.append(SUM_MOD)
if args.all:
	OPERATORS = [FIRST, LAST, MIN, MAX, MED, SUM_MOD]
OPERATORS_ALL = [COPY, FIRST, LAST, MIN, MAX, MED, SUM_MOD]
VALUES = range(10)

VALUE_P = args.p
VALUE_C = args.c

class Node(object):
	def __init__(self, data, depth):
		self.data = data
		self.left = None
		self.right = None
		self.depth = depth
		
class Tree:
	def __init__(self, n=2, max_vals=50, max_depth=1, ref_range=0):
		self.nodes = []
		self.n = n
		self.tree = []
		
		self.max_depth = max_depth
		self.ref_range = ref_range

		self.root = Node([random.choice(OPERATORS)], 0)
		if ref_range is 0:
			self.make_random_tree(self.root, 0)
		else:
			self.make_random_tree_copy(self.root, 0)
		self.get_tree(self.root)
	
	def make_random_tree(self, node, depth):
		""" generate random tree
		"""
		if (depth >= self.max_depth - 1):
			node.right = Node([random.choice(VALUES) for i in range(random.randint(1, args.max_args))], depth+1)
			return
		
		p = random.uniform(0,1)
		if p < VALUE_P:
			node.left = Node([random.choice(OPERATORS)], depth+1)
			self.make_random_tree(node.left, depth+1)
		else:
			node.left = Node([random.choice(VALUES) for i in range(random.randint(1, args.max_args))], depth+1)

		p = random.uniform(0,1)
		if p < VALUE_P:
			node.right = Node([random.choice(OPERATORS)], depth+1)
			self.make_random_tree(node.right, depth+1)
		else:
			node.right = Node([random.choice(VALUES) for i in range(random.randint(1, args.max_args))], depth+1)

	def make_random_tree_copy(self, node, depth):
		""" generate random tree
		"""
		if(depth > self.max_depth - 2):
			c = random.uniform(0,1)
			if c < VALUE_C:
				node.right = Node([COPY, random.randrange(self.ref_range)], depth+1)
			else:
				node.right = Node([random.choice(VALUES) for i in range(random.randint(1, args.max_args))], depth+1)
			return
		
		p = random.uniform(0,1)
		if p < VALUE_P:
			node.left = Node([random.choice(OPERATORS)], depth+1)
			self.make_random_tree_copy(node.left, depth+1)
		else:
			c = random.uniform(0,1)
			if c < VALUE_C:
				node.left = Node([COPY, random.randrange(self.ref_range)], depth+1)
			else:
				node.left = Node([random.choice(VALUES) for i in range(random.randint(1, args.max_args))], depth+1)

		p = random.uniform(0,1)
		if p < VALUE_P:
			node.right = Node([random.choice(OPERATORS)], depth+1)
			self.make_random_tree_copy(node.right, depth+1)
		else:
			c = random.uniform(0,1)
			if c < VALUE_C:
				node.right = Node([COPY, random.randrange(self.ref_range)], depth+1)
			else:
				node.right = Node([random.choice(VALUES) for i in range(random.randint(1, args.max_args))], depth+1)

	def get_tree(self, root):
		def get_tree_(self, node):
			if node.data[0] in OPERATORS:
				self.tree.append(node.data[0])
				if node.left:
					get_tree_(self, node.left)
				if node.right:
					get_tree_(self, node.right)
				self.tree.append(END)
			else:
				for item in node.data:
					self.tree.append(item)
				if node.data[0] is COPY:
					self.tree.append(END)
				if node.left:
					get_tree_(self, node.left)
				if node.right:
					get_tree_(self, node.right)
		
		if not self.tree:
			get_tree_(self, root)

def levelorder(node): 
	if node is None:
		return
	
	queue = []
	result = []
	queue.append(node) 
	while(len(queue) > 0):
		for item in queue[0].data:
			result.append(item)
		node = queue.pop(0)

		if node.left is not None:
			queue.append(node.left)
		if node.right is not None:
			queue.append(node.right)
	return result

def get_val(tree, position, ref_tree=None):
	def to_value(node, ref_tree=None):
		queue = []
		if node.left:
			if node.left.data[0] in VALUES:
				for val in node.left.data:
					queue.append(val)
			elif node.left.data[0] is COPY:
				queue.append(get_val(ref_tree, node.left.data[1])[0])
			else:
				queue.append(to_value(node.left, ref_tree))
		if node.right:
			if node.right.data[0] in VALUES:
				for val in node.right.data:
					queue.append(val)
			elif node.right.data[0] is COPY:
				queue.append(get_val(ref_tree, node.right.data[1])[0])
			else:
				queue.append(to_value(node.right, ref_tree))
		if node.data[0] is SUM_MOD:
			return (np.sum(queue)%10)
		elif node.data[0] is MIN:
			return min(queue)
		elif node.data[0] is MAX:
			return max(queue)
		elif node.data[0] is MED:
			return int(np.median(queue))
		elif node.data[0] is FIRST:
			return queue[0]
		elif node.data[0] is LAST:
			return queue[-1]

	node = tree.root
	count = 0
	queue = []
	queue.append(node)

	while count <= position:
		for val in queue[0].data:
			count += 1
			if count > position:
				if val in OPERATORS:
					val = to_value(queue[0], ref_tree)
				if val is COPY:
					val = get_val(ref_tree, queue[0].data[1])[0]
				return (val, queue[0].depth)
		node = queue.pop(0)
		if node.left is not None:
			queue.append(node.left)
		if node.right is not None:
			queue.append(node.right)


def get_data(tree, position):
	return levelorder(tree.root)[position]

# level order by levels
def levelorder_(root):
	if root is None:
		return []
	result, current = [], [root]
	while current:
		next_level, vals = [], []
		for node in current:
			for item in node.data:
				vals.append(item)
			if node.left:
				next_level.append(node.left)
			if node.right:
				next_level.append(node.right)
		current = next_level
		result.append(vals)
	return result

def sum_level(levelorder_):
	val = []
	for level in levelorder_:
		val.append(np.sum(level))
	return val


def generate_traversal(tree, mode, split_point=2):
	tree = tree.root
	if(mode=='levelorder'):
		order = levelorder(tree)
	elif(mode=='levelorder_'):
		order = levelorder_(tree)
	return order

# print("====================================")
# tree1 = Tree(max_depth=args.max_depth)
# print('tree:  ', '\t', *tree1.tree, sep=' ')
# order1 = generate_traversal(tree1, 'levelorder')
# print('level by level:' , generate_traversal(tree1, 'levelorder_'))
# print('count=', len(order1))
# for i in range(len(order1)):
# 	node = get_val(tree1, i)
# 	print('node ', i, ' ', node[1], ' ', get_data(tree1, i), ' ', node[0])

# print("====================================")
# tree = Tree(max_depth=args.max_depth, ref_range=len(order1))
# print('tree:  ', '\t', *tree.tree, sep=' ')
# order = generate_traversal(tree, 'levelorder')
# print('level by level:' , generate_traversal(tree, 'levelorder_'))
# print('count=', len(order))
# for i in range(len(order)):
# 	node = get_val(tree, i, tree1)
# 	print('node ', i, ' ', node[1], ' ', get_data(tree, i), ' ', node[0])

import copy

def generate_pointers(output, reference):
	output_idx = [reference.index(x) for x in output]
	return output_idx

def generate_dataset(root, name, size, depth, min_depth=0):
	path = root
	
	stats = [0]*len(OPERATORS_ALL)
	stats2 = [0]*len(OPERATORS_ALL)
	length1 = [0]*size
	length2 = [0]*size

	depth1 = [0]*15
	depth2 = [0]*15
	copy = 0
	labels = []
	# generate data file
	counter = 0

	if min_depth != 0:
		depths = range(min_depth, depth + 1)
		n_bins = depth - min_depth + 1
		bin_size = int(size/n_bins) + 1
		bin_counter1 = [0] * (depth+1)
		bin_counter2 = [0] * (depth+1)

	data_path = name + '.input'
	data_path = os.path.join(path, data_path)
	label_path = name + '.label'
	label_path = os.path.join(path, label_path)
	with open(data_path, 'w') as fout:
		while counter<size:
			t1 = Tree(max_depth=depth)
			t1len = len(levelorder(t1.root))
			t1depth = get_val(t1, t1len -1)[1]

			if min_depth == 0:
				if t1depth < depth:
					continue

			t2 = Tree(max_depth=depth, ref_range=t1len)
			t2len = len(levelorder(t2.root))
			t2depth = get_val(t2, t2len -1, t1)[1]

			if min_depth == 0:
				if t2depth < depth:
					continue

			if min_depth != 0:
				if t1depth not in depths or t2depth not in depths:
					continue
				else:
					if bin_counter1[t1depth] == bin_size or bin_counter2[t2depth] == bin_size:
						continue
					else:
						bin_counter1[t1depth] += 1
						bin_counter2[t2depth] += 1

			length1[counter] = t1len
			length2[counter] = t2len
			counter+=1
			depth1[t1depth] += 1
			depth2[t2depth] += 1
			seq1 = [str(x) for x in t1.tree]
			seq2 = [str(x) for x in t2.tree]
			seq = seq1 + ['X'] + seq2
			for x in seq1:
				if x in OPERATORS_ALL:
					stats[OPERATORS_ALL.index(x)] += 1
			
			for x in seq2:
				if x in OPERATORS_ALL:
					stats2[OPERATORS_ALL.index(x)] += 1
			if COPY in seq2:
				copy += 1

			# for i in range(t2len):
			# 	val_ = get_val(t2, i, t1)
			# 	val = [val_[0]]
			# 	print(val)
			val_ = get_val(t1, 0)
			val = [val_[0]]
			val_ = get_val(t2, 0, t1)
			val.append(val_[0])
			# print("====================================")
			# print(t1.tree)
			# print(t2.tree)
			# print('val = ', val)
			labels.append(val)
			fout.write("\t".join([" ".join(seq)]))
			fout.write('\n')

	data_path = 'stats-' + name + '.txt'
	data_path = os.path.join(root, data_path)
	with open(data_path, 'w') as fout:
		stats = stats / np.sum(stats) * 100
		stats2 = stats2 / np.sum(stats2) * 100
		depth1 = depth1 / np.sum(depth1) * 100
		depth2 = depth2 / np.sum(depth2) * 100
		copy = copy / counter * 100
		fout.write('Number of sequences: ')
		fout.write(str(size))
		fout.write('\n')
		fout.write('Operators in tree 1: ')
		fout.write(str(stats))
		fout.write('\n')
		fout.write('Operators in tree 2: ')
		fout.write(str(stats2))
		fout.write('\n')
		fout.write('Average length of tree 1: Mean = ')
		fout.write(str(np.average(length1)))
		fout.write('\t Stdev = ')
		fout.write(str(np.std(length1)))
		fout.write('\n')
		fout.write('Average length of tree 2: Mean = ')
		fout.write(str(np.average(length2)))
		fout.write('\t Stdev = ')
		fout.write(str(np.std(length2)))
		fout.write('\n')
		fout.write('Depths in tree 1: ')
		fout.write(str(depth1))
		fout.write('\n')
		fout.write('Depths in tree 2: ')
		fout.write(str(depth2))
		fout.write('\n')
		fout.write('Percentage of trees with copy operator: ')
		fout.write(str(copy))

	with open(label_path, 'w') as fout:
		for label in labels:
			b = [str(x) for x in label]
			fout.write(",".join(b))
			fout.write('\n')


if __name__ == '__main__':
	toy_dir = args.dir
	if not os.path.exists(toy_dir):
		os.mkdir(toy_dir)
	
	generate_dataset(toy_dir, 'train', 1000 * args.size, args.max_depth, min_depth=3)
	generate_dataset(toy_dir, 'valid', 100 * args.size, args.max_depth, min_depth=3)
	generate_dataset(toy_dir, 'test', 100 * args.size, args.max_depth, min_depth=3)

	for i in range(3,13):
		generate_dataset(toy_dir, 'test{}'.format(i), 100 * args.size, i)

	# generate_dataset(toy_dir, 'test', 10, args.max_depth)
