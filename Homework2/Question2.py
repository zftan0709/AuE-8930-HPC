class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
def solution(root):
    if(root == None):
        return 0;
    if(root.left==None and root.right==None):
        return 0;
    else:
        return max(solution(root.left),solution(root.right))+1;
    return None;

a15=TreeNode(15)
a7=TreeNode(7)
a20=TreeNode(20)
a9=TreeNode(9)
a3=TreeNode(3)
a5 = TreeNode(2)
a1=TreeNode(5)
a4=TreeNode(11)

a20.left=a15
a20.right=a7
a3.left=a9
a3.right=a20
a15.right=a5
a9.left=a1
a9.right=a4

print(solution(a3))