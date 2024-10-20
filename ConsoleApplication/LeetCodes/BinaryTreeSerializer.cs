using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeetCodes
{
    public class BinaryTreeSerializer
    {
        // Encodes a tree to a single string.
        public string serialize(TreeNode root)
        {
            if (root == null) return "";
            var q = new Queue<TreeNode>();
            StringBuilder result = new StringBuilder();
            q.Enqueue(root);
            while (q.Count > 0)
            {
                TreeNode temp = q.Dequeue();
                if (temp == null)
                {
                    result.Append("null ");
                }
                else
                {
                    result.Append(temp.val + " ");
                    q.Enqueue(temp.left);
                    q.Enqueue(temp.right);
                }
            }
            return result.ToString();
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(string data)
        {
            if (data == "") return null;
            string[] arr = data.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var q = new Queue<TreeNode>();

            TreeNode tree = new TreeNode(Int32.Parse(arr[0]));
            q.Enqueue(tree);
            for (int i = 1; i < arr.Length; i++)
            {
                var parent = q.Dequeue();
                if (arr[i] != "null")
                {
                    TreeNode left = new TreeNode(Int32.Parse(arr[i]));
                    parent.left = left;
                    q.Enqueue(left);
                }
                if (arr[++i] != "null")
                {
                    TreeNode right = new TreeNode(Int32.Parse(arr[i]));
                    parent.right = right;
                    q.Enqueue(right);
                }
            }
            return tree;
        }
    }

    public class TreeNode
    {
        public int val;
        public TreeNode left;
        public TreeNode right;
        public TreeNode(int x) { val = x; }
    }
}
