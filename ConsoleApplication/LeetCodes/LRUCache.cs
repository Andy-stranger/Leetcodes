using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeetCodes
{
    public class LRUCache
    {

        public int limit;
        public Node head;
        public Node tail;
        public Dictionary<int, Node> map;

        public LRUCache(int capacity)
        {
            limit = capacity;
            tail = new Node(0, 0);
            head = new Node(0, 0);
            head.next = tail;
            tail.prev = head;
            map = new Dictionary<int, Node>();
        }

        public int Get(int key)
        {
            if (!map.ContainsKey(key)) return -1;
            Node valNode = map[key];
            var val = valNode.value;
            DeleteElement(valNode);
            InsertElement(valNode);
            return val;
        }

        public void Put(int key, int value)
        {
            if (map.ContainsKey(key)) DeleteElement(map[key]);
            if (map.Count == limit) DeleteElement(tail.prev);
            InsertElement(new Node(key, value));
        }

        private void DeleteElement(Node node)
        {
            map.Remove(node.key);
            node.prev.next = node.next;
            node.next.prev = node.prev;
        }

        private void InsertElement(Node node)
        {
            var headNext = head.next;
            head.next = node;
            node.next = headNext;
            headNext.prev = node;
            node.prev = head;
            map.Add(node.key, node);
        }
    }

    public class Node
    {
        public int key;
        public int value;
        public Node prev;
        public Node next;
        public Node(int k, int v)
        {
            key = k;
            value = v;
            prev = null;
            next = null;
        }
    }

    /**
     * Your LRUCache object will be instantiated and called as such:
     * LRUCache obj = new LRUCache(capacity);
     * int param_1 = obj.Get(key);
     * obj.Put(key,value);
     */
}
