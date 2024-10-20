using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeetCodes
{
    public class LFUCache
    {

        public int limit;
        public Node head;
        public Node tail;
        public Dictionary<int, Node> cache_map;
        public Dictionary<int, int> use_counter;

        public LFUCache(int capacity)
        {
            limit = capacity;
            tail = new Node(0, 0);
            head = new Node(0, 0);
            head.next = tail;
            tail.prev = head;
            cache_map = new Dictionary<int, Node>();
            use_counter = new Dictionary<int, int>();
        }

        public int Get(int key)
        {
            if (!cache_map.ContainsKey(key)) return -1;
            Node valNode = cache_map[key];
            var val = valNode.value;
            DeleteElement(valNode);
            InsertElement(valNode);
            use_counter[key]++;
            return val;
        }

        public void Put(int key, int value)
        {
            if (cache_map.ContainsKey(key)) DeleteElement(cache_map[key]);
            if (cache_map.Count == limit)
            {
                var min = use_counter.Values.Min();
                var leastFreq = use_counter
                                .Where(pair => pair.Value == min)
                                .Select(pair => pair.Key)
                                .ToList();
                if (leastFreq.Count == 0) DeleteElement(cache_map[leastFreq.First()]);
                else
                {
                    var temp = tail;
                    while (temp != head)
                    {
                        if (leastFreq.Contains(temp.key)) break;
                        temp = temp.prev;
                    }
                    DeleteElement(temp);
                }
            }
            InsertElement(new Node(key, value));
            if (!use_counter.ContainsKey(key)) use_counter.Add(key, 0);
            use_counter[key]++;
        }

        private void DeleteElement(Node node)
        {
            cache_map.Remove(node.key);
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
            cache_map.Add(node.key, node);
        }
    }

    /**
     * Your LFUCache object will be instantiated and called as such:
     * LFUCache obj = new LFUCache(capacity);
     * int param_1 = obj.Get(key);
     * obj.Put(key,value);
     */
}
