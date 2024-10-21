using LeetCodes;
using LeetCodes.Sorting;
using Newtonsoft.Json;
using System.Collections.Generic;
using System.Text;
using static System.Runtime.InteropServices.JavaScript.JSType;

public class Solution
{
    public static int timer = 1;
    public static void Main()
    {
        /*
        //Console.WriteLine("Give the string");
        //var str = Console.ReadLine();
        //if (str == null) return;
        //RemoveKdigits(str, 1);

        //SumSubarrayMins([3, 1, 2, 4]);

        //AsteroidCollision([-2, -1, 1, 2]);

        //MinCoins([9, 2, 11, 5, 14, 17, 8, 18], 8, 39);

        //int[] array = [5, 1, 2, 3, 4];
        //var sorter = new MergeSort();
        //sorter.Sort(array, 0, array.Length-1);
        //Console.WriteLine("done");

        //Job[] jobs =
        //{
        //    new(1,4,20),
        //    new(2,1,10),
        //    new(3,1,40),
        //    new(4,1,30)
        //};

        //JobScheduling(jobs, 4);

        //CountOfPeaks([4, 1, 4, 2, 1, 5], [[2, 2, 4], [1, 0, 2], [1, 0, 4]]);

        //var LRU_cache = new LRUCache(2);
        //LRU_cache.Put(1,1);
        //LRU_cache.Put(2,2);
        //LRU_cache.Get(1);
        //LRU_cache.Put(3,3);

        //var LFU_cache = new LFUCache(2);
        //LFU_cache.Put(2, 2);
        //LFU_cache.Put(1, 1);
        //var one = LFU_cache.Get(2);
        //var two = LFU_cache.Get(1);
        //var three = LFU_cache.Get(2);
        //LFU_cache.Put(3, 3);
        //LFU_cache.Put(4, 4);
        //var four = LFU_cache.Get(3);
        //var five = LFU_cache.Get(2);
        //var six = LFU_cache.Get(1);
        //var seven = LFU_cache.Get(4);
        //List<int> list = [one, two, three, four, five, six, seven];
        //foreach(var num in list) Console.WriteLine(num);
        //Console.ReadLine();

        //MaxProfitAssignment([2, 4, 6, 8, 10], [10, 20, 30, 40, 50], [4, 5, 6, 7]);

        //FindMaximizedCapital(3, 0, [1, 2, 3], [0, 1, 2]);
        */

        #region old code
        //MaxSatisfied_Optimal([1, 0, 1, 2, 1, 1, 7, 5], [0, 1, 0, 1, 0, 1, 0, 1], 3);

        //LongestSubarray([8, 2, 4, 7], 4);

        //var result = TimeTaken([[0, 1], [0, 2]]);
        //Console.WriteLine(JsonConvert.SerializeObject(result.ToList()));

        //var result = LadderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]);
        //Console.WriteLine(result); 

        //var res = ShortestPath(5, 6, [[1, 2, 2], [2, 5, 5], [2, 3, 4], [1, 4, 1], [4, 3, 3], [3, 5, 1]]);
        //foreach(var v in res) Console.WriteLine(v);
        //Console.ReadLine();

        //var res = ShortestPathBinaryMatrix([[0]]);
        //Console.WriteLine(res);

        //var res = MinimumEffortPath([[1, 2, 2], [3, 8, 2], [5, 3, 5]]);
        //Console.WriteLine(res);

        //var res = FindCheapestPrice(4, [[0, 1, 100], [1, 2, 100], [2, 0, 100], [1, 3, 600], [2, 3, 200]], 0, 3, 1);
        //Console.WriteLine(res);

        //var res = NetworkDelayTime([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2);
        //Console.WriteLine(res);

        //var res = ArticulationPoints(9, [[3, 4], [1, 3], [1, 4], [6, 6], [5, 5], [1, 2], [2, 2], [0, 2], [2, 6], [1, 5], [2, 6], [6, 3], [8, 3], [2, 2], [6, 6], [6, 5], [1, 5], [2, 0], [6, 2], [2, 8], [3, 1], [8, 6], [1, 0], [1, 6], [3, 0], [4, 8], [3, 4], [6, 5], [6, 0], [8, 0], [2, 3], [7, 6]]);
        //foreach (var a in res) Console.WriteLine(a);
        #endregion

        //var res = Kosaraju(5, [[1, 0], [0, 2], [2, 1], [0, 3], [3, 4]]);
        //Console.WriteLine(res); 

        //var ans = CoinChange([1, 2, 5], 11);
        //Console.WriteLine(ans);

        //var res = LongestStrChain(["a", "b", "ba", "bca", "bda", "bdca"]);
        //Console.WriteLine(res);
        //string val = "myString";
        //var s = val.Hash(53, (int)1e9+9);
        //Console.WriteLine(s);

        int[] arr = [5,4,3,2,1];
        foreach(var i in arr) Console.Write(i + " ");
        Console.WriteLine();
        arr.BubbleSort();
        foreach (var i in arr) Console.Write(i + " ");
        Console.ReadLine();
    }

    public static int LongestStrChain(string[] words)
    {
        Array.Sort(words, (a, b) => a.Length - b.Length);
        return longestIncreasingSubsequence(words.Length, words);
    }

    private static int longestIncreasingSubsequence(int n, string[] words)
    {
        var dp = new int[n];
        var max = 1;
        for (int i = 0; i < n; i++)
        {
            dp[i] = 1;
            for (int j = 0; j < i; j++)
            {
                if (IsChain(words[i], words[j]) && dp[i] < 1 + dp[j])
                {
                    dp[i] = 1 + dp[j];
                }
            }
            if (max < dp[i])
            {
                max = dp[i];
            }
        }
        // IList<int> list = new List<int>();
        // list.Add(arr[maxInd]);
        // while(bt[maxInd] != maxInd){
        //     maxInd = bt[maxInd];
        //     list.Add(arr[maxInd]);
        // }
        return max;
    }

    private static bool IsChain(string str1, string str2)
    {
        if (str1.Length != str2.Length + 1) return false;
        int first = 0;
        int second = 0;
        while (first < str1.Length)
        {
            if (str1[first] == str2[second])
            {
                first++;
                second++;
            }
            else first++;
        }
        return first == str1.Length && second == str2.Length;
    }

    public static int CoinChange(int[] coins, int amount)
    {
        var dp = new List<List<int>>();
        for (int i = 0; i < coins.Length; i++)
        {
            var list = new List<int>();
            for (int j = 0; j <= amount; j++) list.Add(-1);
            dp.Add(list);
        }
        return fun(coins.Length - 1, amount, coins, dp);
    }

    private static int fun(int ind, int tar, int[] arr, List<List<int>> dp)
    {
        if (ind == 0)
        {
            if (tar % arr[ind] == 0) return tar / arr[ind];
            else return (int)1e9;
        }
        if (dp[ind][tar] != -1) return dp[ind][tar];
        var not_take = fun(ind - 1, tar, arr, dp);
        var take = int.MaxValue;
        if (arr[ind] <= tar) take = 1 + fun(ind, tar - arr[ind], arr, dp);
        dp[ind][tar] = Math.Min(not_take, take);
        return dp[ind][tar];
    }

    //Leetcode 402. Remove K Digits
    public static string RemoveKdigits(string num, int k)
    {
        Stack<char> stack = new Stack<char>();
        char[] chars = num.ToCharArray();
        for (int i = 0; i < chars.Length; i++)
        {
            while (stack.Count > 0 && stack.Peek() > chars[i] && k > 0)
            {
                stack.Pop();
                k--;
            }
            if (stack.Count == 0 && chars[i] == '0') continue;
            stack.Push(chars[i]);
        }
        while (k > 0 && stack.Count > 0)
        {
            stack.Pop();
            k--;
        }
        if (stack.Count == 0) return "0";
        StringBuilder result = new StringBuilder();
        while (stack.Count > 0)
        {
            result.Append(stack.Peek().ToString());
            stack.Pop();
        }
        return new string(result.ToString().Reverse().ToArray());
    }

    public static int SumSubarrayMins(int[] arr)
    {
        int totalSubArrays = 1 << (arr.Length);
        int sum = 0;
        List<List<int>> list = new List<List<int>>();
        for (int i = 0; i < totalSubArrays; i++)
        {
            List<int> current = new();
            for (int j = 0; j < arr.Length; j++)
            {
                if ((i & (1 << j)) != 0) current.Add(arr[j]);
            }
            if (current.Any()) sum += current.Min();
            list.Add(current);
        }
        return sum;
    }

    public static int[] AsteroidCollision(int[] asteroids)
    {
        Stack<int> stack = new();
        for (int i = 0; i < asteroids.Length; i++)
        {
            while (stack.Count > 0 && (stack.Peek() > 0 && asteroids[i] < 0))
            {
                if (Math.Abs(stack.Peek()) < Math.Abs(asteroids[i]))
                {
                    stack.Pop();
                }
                else if (Math.Abs(stack.Peek()) > Math.Abs(asteroids[i]))
                {
                    i++;
                }
                else
                {
                    stack.Pop();
                    i++;
                }
                if (i >= asteroids.Length) break;
            }
            if (i >= asteroids.Length) break;
            stack.Push(asteroids[i]);
        }
        var arr = stack.ToArray();
        Array.Reverse(arr);
        return arr;
    }

    //Greedy algo, 
    //Problem: Number of Coins - GFG
    public static int MinCoins(int[] coins, int M, int V)
    {
        Array.Sort(coins);
        Array.Reverse(coins);
        int count = 0;
        int value = 0;
        for (int i = 0; i < M; i++)
        {
            if (coins[i] <= V - value)
            {
                var req_num = (V - value) / coins[i];
                value += coins[i] * req_num;
                count++;
            }
            if (value == V) return count;
        }
        return -1;
    }

    //Greedy - Job scheduling algo - GFG
    public static int[] JobScheduling(Job[] arr, int n)
    {
        arr = arr.ToList().OrderByDescending(a => a.profit).ToArray();
        int maxDead = 0;
        foreach (var job in arr) if (job.dead > maxDead) maxDead = job.dead;
        int[] jobArray = new int[maxDead + 1];
        int maxJob = 0;
        int maxProfit = 0;
        for (int i = 0; i < n; i++)
        {
            Job currentJob = arr[i];
            int j = currentJob.dead;
            while (j > 0 && jobArray[j] > 0)
            {
                j--;
            }
            if (j > 0)
            {
                jobArray[j] = currentJob.id;
                maxJob++;
                maxProfit += currentJob.profit;
            }
        }
        return new int[] { maxJob, maxProfit };
    }

    //Leetcode contest question - 16/6/2024
    #region Leetcode contest question - 16/6/2024
    public static int CountCompleteDayPairs(int[] hours)
    {

        int[] remainderCounts = new int[24];

        int count = 0;

        foreach (int hour in hours)
        {

            int remainder = hour % 24;

            // The complement remainder we need to form a complete day

            int complement = (24 - remainder) % 24;

            // Add the count of numbers that can pair with this one

            count += remainderCounts[complement];

            // Increment the count for this remainder

            remainderCounts[remainder]++;

        }

        return count;
    }
    public static long MaximumTotalDamage(int[] power)
    {

        // Frequency map to count the occurrences of each damage value

        Dictionary<int, int> freqMap = new();

        foreach (int p in power)
        {
            if (!freqMap.ContainsKey(p)) freqMap.Add(p, 0);
            freqMap[p]++;
        }



        // Extract unique damage values and sort them

        List<int> uniquePowers = [.. freqMap.Keys];
        uniquePowers = uniquePowers.OrderBy(p => p).ToList();



        // DP array to store maximum damage up to each unique damage value

        int n = uniquePowers.Count;

        long[] dp = new long[n];



        // Initialize the dp array

        dp[0] = (long)uniquePowers[0] * freqMap[uniquePowers[0]];



        for (int i = 1; i < n; i++)
        {

            int currentPower = uniquePowers[i];

            long currentDamage = (long)currentPower * freqMap[currentPower];



            // Start with the previous maximum value

            dp[i] = dp[i - 1];



            // Check if we can take this power by looking for the nearest valid power

            int j = i - 1;

            while (j >= 0 && uniquePowers[j] >= currentPower - 2)
            {

                j--;

            }



            if (j >= 0)
            {

                dp[i] = Math.Max(dp[i], currentDamage + dp[j]);

            }
            else
            {

                dp[i] = Math.Max(dp[i], currentDamage);

            }

        }



        // The answer is the maximum value in the dp array

        return dp[n - 1];

    }

    class FenwickTree
    {
        private int[] tree;
        public FenwickTree(int size)
        {

            tree = new int[size + 1];

        }
        public void update(int index, int delta)
        {

            index++; // Fenwick Tree is 1-indexed

            while (index < tree.Length)
            {

                tree[index] += delta;

                index += index & -index;

            }

        }
        public int query(int index)
        {

            index++; // Fenwick Tree is 1-indexed

            int sum = 0;

            while (index > 0)
            {

                sum += tree[index];

                index -= index & -index;

            }

            return sum;

        }
        public int rangeQuery(int left, int right)
        {

            return query(right) - query(left - 1);

        }
    }

    public static List<int> CountOfPeaks(int[] nums, int[][] queries)
    {

        int n = nums.Length;

        bool[] isPeak = new bool[n];

        FenwickTree fenwickTree = new FenwickTree(n);



        // Initialize peaks and Fenwick Tree

        for (int i = 1; i < n - 1; i++)
        {

            if (nums[i] > nums[i - 1] && nums[i] > nums[i + 1])
            {

                isPeak[i] = true;

                fenwickTree.update(i, 1);

            }

        }



        List<int> result = new();



        foreach (int[] query in queries)
        {

            if (query[0] == 1)
            {

                int li = query[1];

                int ri = query[2];

                if (li == ri || li + 1 == ri)
                {

                    result.Add(0);

                }
                else
                {

                    result.Add(fenwickTree.rangeQuery(li + 1, ri - 1));

                }

            }
            else if (query[0] == 2)
            {

                int index = query[1];

                int val = query[2];

                nums[index] = val;



                // Update peaks and Fenwick Tree for the current index and its neighbors

                for (int i = Math.Max(1, index - 2); i <= Math.Min(n - 2, index + 2); i++)
                {

                    bool wasPeak = isPeak[i];

                    bool nowPeak = nums[i] > nums[i - 1] && nums[i] > nums[i + 1];

                    if (wasPeak != nowPeak)
                    {

                        isPeak[i] = nowPeak;

                        fenwickTree.update(i, nowPeak ? 1 : -1);

                    }

                }

            }

        }



        return result;

    }
    #endregion

    public static int MaxProfitAssignment(int[] difficulty, int[] profit, int[] worker)
    {
        worker = worker.OrderByDescending(a => a).ToList().ToArray();
        var profit_difficulty = profit
                                .Zip(difficulty, (p, d) => new KeyValuePair<int, int>(p, d))
                                .OrderByDescending(a => a.Key).ToList();
        int pInd = 0;
        int total_profit = 0;
        int wInd = 0;
        while (wInd < worker.Length && pInd < profit_difficulty.Count)
        {
            if (profit_difficulty[pInd].Value <= worker[wInd])
            {
                total_profit += profit_difficulty[pInd].Key;
                wInd++;
            }
            else pInd++;
        }
        return total_profit;
    }

    public static int FindMaximizedCapital(int k, int w, int[] profits, int[] capital)
    {
        var profit_capital = profits.Zip(capital, (p, c) => new KeyValuePair<int, int>(p, c)).OrderByDescending(a => a.Key).ToList();
        int i = 0;
        Queue<KeyValuePair<int, int>> q = new();
        while (i < profit_capital.Count && k > 0)
        {
            if (profit_capital[i].Value <= w)
            {
                var currentProfit = profit_capital[i].Key;
                if (q.Count > 0 && q.Peek().Value <= w) currentProfit = Math.Max(currentProfit, q.Peek().Key);
                w += currentProfit;
                k--;
            }
            else q.Enqueue(profit_capital[i]);
            i++;
        }
        while (k > 0)
        {
            if (q.Count == 0) break;
            if (q.Peek().Value <= w)
            {
                w += q.Dequeue().Key;
                k--;
            }
            else
            {
                Stack<KeyValuePair<int, int>> s = new();
                while (q.Peek().Value > w) s.Push(q.Dequeue());
                var deq = q.Dequeue();
                while (s.Count > 0) q.Enqueue(s.Pop());
                q.Enqueue(deq);
            }
        }
        return w;
    }

    public static int MaxSatisfied_Better(int[] customers, int[] grumpy, int minutes)
    {
        int maxCustomers = 0;
        int left = 0;
        int right = minutes - 1;
        for (int j = 0; j < grumpy.Length; j++) if (grumpy[j] == 0) maxCustomers += customers[j];
        List<int> grumpyMinutes = [];
        for (int i = 0; i < grumpy.Length; i++) if (grumpy[i] == 1) grumpyMinutes.Add(i);
        int lastGrumpy = 0;
        while (right < grumpy.Length)
        {
            var grumpySum = grumpyMinutes.Where(a => a >= left && a <= right).Select(a => customers[a]).Sum();
            if (maxCustomers < maxCustomers + grumpySum - lastGrumpy)
            {
                maxCustomers = maxCustomers + grumpySum - lastGrumpy;
                lastGrumpy = grumpySum;
            }
            left++;
            right++;
        }
        return maxCustomers;
    }

    public static int MaxSatisfied_Optimal(int[] customers, int[] grumpy, int minutes)
    {
        int maxCustomers = 0;
        int left = 0;
        int right = minutes - 1;
        for (int j = 0; j < grumpy.Length; j++) if (grumpy[j] == 0) maxCustomers += customers[j];
        int tempSum = 0;
        for (int k = 0; k < minutes; k++) if (grumpy[k] == 1) tempSum += customers[k];
        left++;
        right++;
        int currentSum = tempSum;
        while (right < grumpy.Length)
        {
            if (grumpy[left - 1] == 1) currentSum -= customers[left - 1];
            if (grumpy[right] == 1) currentSum += customers[right];
            tempSum = Math.Max(currentSum, tempSum);
            left++;
            right++;
        }
        return maxCustomers + tempSum;
    }

    public static int LongestSubarray(int[] nums, int limit)
    {
        int maxLength = 1;
        int min = nums[0];
        int max = nums[0];
        int left = 0;
        int right = 1;
        while (left <= right && right < nums.Length)
        {
            min = Math.Min(min, Math.Min(nums[left], nums[right]));
            max = Math.Max(max, Math.Max(nums[left], nums[right]));
            if (max - min <= limit)
            {
                maxLength = Math.Max(maxLength, right - left + 1);
                right++;
            }
            else left++;
        }
        return maxLength;
    }

    public static long MaximumImportance(int n, int[][] roads)
    {
        Dictionary<int, int> map = new();
        foreach (var vertices in roads)
        {
            if (!map.ContainsKey(vertices[0])) map.Add(vertices[0], 0);
            if (!map.ContainsKey(vertices[1])) map.Add(vertices[1], 0);
            map[vertices[0]]++;
            map[vertices[1]]++;
        }
        map = map.OrderByDescending(a => a.Value).ToDictionary();
        foreach (var kvp in map)
        {
            map[kvp.Key] = n;
            n--;
            if (n == 0) break;
        }
        long maxImportance = 0;
        foreach (var vertices in roads)
        {
            maxImportance += (long)map[vertices[0]] + map[vertices[1]];
        }
        return maxImportance;
    }

    public static int[] TimeTaken(int[][] edges)
    {
        int[] times = new int[edges.Length + 1];
        List<int>[] adj = new List<int>[edges.Length + 1];
        for (int i = 0; i < edges.Length + 1; i++) adj[i] = new List<int>();
        for (int j = 0; j < edges.Length; j++)
        {
            if (!adj[edges[j][0]].Contains(edges[j][1])) adj[edges[j][0]].Add(edges[j][1]);
        }
        for (int i = 0; i < edges.Length + 1; i++)
        {
            Queue<(int node, List<int> edg)> q = new();
            int[] vis = new int[edges.Length + 1];
            var edge = adj[i];
            q.Enqueue((i, edge));
            vis[i] = 1;
            var time = 0;
            while (q.Count > 0)
            {
                var (n, e) = q.Dequeue();
                if (n % 2 == 0) time += 2;
                else time += 1;
                foreach (var node in e)
                {
                    if (vis[node] != 1)
                    {
                        q.Enqueue((node, adj[node]));
                    }
                }
            }
            times[i] = time;
        }
        return times;
    }

    public int LadderLength(string beginWord, string endWord, IList<string> wordList)
    {
        if (!wordList.Contains(endWord)) return 0;
        Queue<(int step, string word)> q = new();
        var hashSet = new HashSet<string>(wordList);
        q.Enqueue((1, beginWord));
        hashSet.Remove(beginWord);
        while (q.Count > 0)
        {
            var (steps, wrd) = q.Dequeue();
            if (wrd == endWord) return steps;
            foreach (var word in hashSet)
            {
                if (DiffOneChar(wrd, word))
                {
                    q.Enqueue((steps + 1, word));
                    hashSet.Remove(word);
                }
            }
        }
        return 0;
    }
    private bool DiffOneChar(string word1, string word2)
    {
        int count = 0;
        for (int i = 0; i < word1.Length; i++)
        {
            if (word1[i] != word2[i]) count++;
        }
        return count == 1;
    }

    public static List<int> dijkstra(int V, List<List<int>>[] adj, int S)
    {
        PriorityQueue<Pair, int> pq = new PriorityQueue<Pair, int>();
        int[] distance = new int[V];
        for (int i = 0; i < V; i++) distance[i] = int.MaxValue;
        distance[S] = 0;
        pq.Enqueue(new Pair(0, S), 0);
        while (pq.Count > 0)
        {
            var pr = pq.Dequeue();
            var currDist = pr.dist;
            var currNode = pr.node;
            for (int i=0; i< adj[currNode].Count; i++)
            {
                var edgeWt = adj[currNode][i][1];
                var adjNode = adj[currNode][i][0];
                if (distance[adjNode] > currDist + edgeWt)
                {
                    distance[adjNode] = currDist + edgeWt;
                    pq.Enqueue(new Pair(distance[adjNode], adjNode), distance[adjNode]);
                    
                }
            }
        }
        return [.. distance];
    }

    public static List<int> ShortestPath(int n, int m, int[][] edges)
    {
        List<List<int>>[] adj = new List<List<int>>[n+1];
        for (int i = 1; i<=n; i++) adj[i] = new List<List<int>>();
        foreach(var arr in edges)
        {
            adj[arr[0]].Add(new List<int>() { arr[1], arr[2] });
            adj[arr[1]].Add(new List<int>() { arr[0], arr[2] });
        }
        int V = n;
        int S = 1;
        int[] parent = new int[n + 1];
        int[] distance = new int[V + 1];
        PriorityQueue<Pair, int> pq = new();
        for (int i = 1; i <= V; i++)
        {
            distance[i] = int.MaxValue;
            parent[i] = i;
        }
        distance[S] = 0;
        pq.Enqueue(new Pair(0, S), 0);
        while (pq.Count > 0)
        {
            var pr = pq.Dequeue();
            var currDist = pr.dist;
            var currNode = pr.node;
            for (int i = 0; i < adj[currNode].Count; i++)
            {
                var edgeWt = adj[currNode][i][1];
                var adjNode = adj[currNode][i][0];
                if (distance[adjNode] > currDist + edgeWt)
                {
                    distance[adjNode] = currDist + edgeWt;
                    pq.Enqueue(new Pair(distance[adjNode], adjNode), distance[adjNode]);
                    parent[adjNode] = currNode;
                }
            }
        }
        if (distance[n] == int.MaxValue) return [-1];
        var res = new List<int>();
        int node = n;
        while (parent[node] != node)
        {
            res.Add(node);
            node = parent[node];
        }
        res.Add(S);
        res.Reverse();
        return res;
    }

    public static int ShortestPathBinaryMatrix(int[][] grid)
    {
        var n = grid.Length;
        int[,] path = new int[n, n];
        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < n; c++)
            {
                path[r, c] = int.MaxValue;
            }
        }
        Queue<Data> q = new();
        int[] delRow = [-1, -1, -1, 0, 0, 1, 1, 1];
        int[] delCol = [-1, 0, 1, -1, 1, -1, 0, 1];
        if (grid[0][0] == 1) return -1;
        int i = 0;
        int j = 0;
        path[i, j] = 1;
        q.Enqueue(new Data(1, 0, 0));
        while (q.Count > 0)
        {
            var data = q.Dequeue();
            var currDist = data.dist;
            var currRow = data.row;
            var currCol = data.col;
            for (int k = 0; k < 8; k++)
            {
                var row = currRow + delRow[k];
                var col = currCol + delCol[k];
                if (row >= 0 && row < n && col >= 0 && col < n && grid[row][col] == 0 && path[row, col] > currDist + 1)
                {
                    q.Enqueue(new Data(currDist + 1, row, col));
                    path[row, col] = currDist + 1;
                }
                if (row == n - 1 && col == n - 1 && path[row, col] != int.MaxValue) return path[row, col];
            }
        }
        return path[n - 1, n - 1] == int.MaxValue ? -1 : path[n - 1, n - 1];
    }

    public static int MinimumEffortPath(int[][] heights)
    {
        int maxDiff = int.MaxValue;
        int rowLen = heights.Length;
        int colLen = heights[0].Length;
        PriorityQueue<Data, int> pq = new();
        int[] delta = [-1, 0, 1, 0, -1];
        pq.Enqueue(new Data(heights[0][0], 0, 0), 0);
        while (pq.Count > 0)
        {
            var data = pq.Dequeue();
            var currHeight = data.dist;
            var currRow = data.row;
            var currCol = data.col;
            for (int i = 0; i < 4; i++)
            {
                var newRow = currRow + delta[i];
                var newCol = currCol + delta[i + 1];
                if (newRow >= 0 && newRow < rowLen
                && newCol >= 0 && newCol < colLen
                && Math.Abs(currHeight - heights[newRow][newCol]) < maxDiff)
                {
                    pq.Enqueue(new Data(heights[newRow][newCol], newRow, newCol), Math.Abs(currHeight - heights[newRow][newCol]));
                    maxDiff = Math.Abs(currHeight - heights[newRow][newCol]);
                }
            }
        }
        return maxDiff;
    }
    public static int FindCheapestPrice(int n, int[][] flights, int src, int dst, int k)
    {
        int edges = flights.Length;
        List<List<int>>[] adj = new List<List<int>>[n];
        for (int i = 0; i < n; i++) adj[i] = new List<List<int>>();
        foreach (var arr in flights) adj[arr[0]].Add(new List<int> { arr[1], arr[2] });
        int[] dist = new int[n];
        for (int i = 0; i < n; i++) dist[i] = int.MaxValue;
        dist[src] = 0;
        PriorityQueue<Tripplet, int> pq = new();
        pq.Enqueue(new Tripplet(0, src, -1), 0);
        while (pq.Count > 0)
        {
            var curr = pq.Dequeue();
            var currPrice = curr.price;
            var currNode = curr.node;
            var currStops = curr.stops;
            foreach (var list in adj[currNode])
            {
                var dest = list.First();
                var price = list.Last();
                if (dist[dest] > price + currPrice)
                {
                    dist[dest] = price + currPrice;
                    pq.Enqueue(new Tripplet(dist[dest], dest, currStops + 1), dist[dest]);
                }
                if (dest == dst && currStops + 1 >= k) return dist[dest];
            }
        }
        return -1;
    }

    public static int NetworkDelayTime(int[][] times, int n, int k)
    {
        List<List<int>>[] adj = new List<List<int>>[n + 1];
        for (int i = 1; i <= n; i++) adj[i] = new List<List<int>>();
        foreach (var arr in times) adj[arr[0]].Add(new List<int> { arr[1], arr[2] });
        int[] time = new int[n + 1];
        for (int i = 1; i <= n; i++) time[i] = int.MaxValue;
        time[0] = -1;
        PriorityQueue<Pair, int> pq = new();
        pq.Enqueue(new Pair(0, k), 0);
        time[k] = 0;
        while (pq.Count > 0)
        {
            var curr = pq.Dequeue();
            var currTime = curr.dist;
            var currNode = curr.node;
            foreach (var list in adj[currNode])
            {
                var newNode = list.First();
                var newTime = list.Last();
                if (time[newNode] > newTime + currTime)
                {
                    time[newNode] = newTime + currTime;
                    pq.Enqueue(new Pair(time[newNode], newNode), time[newNode]);
                }
            }
        }
        return time.ToList().All(a => a != int.MaxValue) ? time.ToList().Max() : -1;
    }

    public static List<int> ArticulationPoints(int n, List<int>[] adj)
    {
        int[] vis = new int[n];
        int[] timeIn = new int[n];
        int[] low = new int[n];
        int[] mark = new int[n];
        List<int> result = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (vis[i] == 0) Dfs(i, -1, vis, timeIn, low, adj, mark);
        }
        for (int i = 0; i < n; i++) if (mark[i] == 1) result.Add(i);
        if (result.Count == 0) return new List<int> { -1 };
        return result;
    }
    private static void Dfs(int node, int parent, int[] vis, int[] timeIn, int[] low, List<int>[] adj, int[] mark)
    {
        vis[node] = 1;
        timeIn[node] = timer;
        low[node] = timer;
        timer++;
        int child = 0;
        foreach (var a in adj[node])
        {
            if (a != parent)
            {
                if (vis[a] == 0)
                {
                    Dfs(a, node, vis, timeIn, low, adj, mark);
                    low[node] = Math.Min(low[node], low[a]);
                    if (low[a] >= timeIn[node] && parent != -1) mark[node] = 1;
                    child++;
                }
                else
                {
                    low[node] = Math.Min(low[node], timeIn[a]);
                }
            }
        }
        if (child > 1 && parent == -1) mark[node] = 1;
    }

    private static void InitialDfs(int node, int[] vis, List<int>[] adj, Stack<int> st)
    {
        vis[node] = 1;
        foreach (var a in adj[node])
        {
            if (vis[a] == 0) InitialDfs(a, vis, adj, st);
        }
        st.Push(node);
    }
    private static void LaterDfs(int node, int[] vis, List<int>[] adj)
    {
        vis[node] = 1;
        foreach (var a in adj[node])
        {
            if (vis[a] == 0) LaterDfs(a, vis, adj);
        }
    }
    public static int Kosaraju(int V, List<List<int>> adj)
    {
        try
        {
            int[] vis = new int[V];
            List<int>[] adjList = new List<int>[V];
            List<int>[] adjTList = new List<int>[V];
            for (int i = 0; i < V; i++)
            {
                adjList[i] = new List<int>();
                adjTList[i] = new List<int>();
            }
            foreach (var arr in adj)
            {
                var from = arr[0];
                var to = arr[1];
                adjList[from].Add(to);
                adjTList[to].Add(from);
            }
            Stack<int> st = new Stack<int>();
            for (int i = 0; i < V; i++)
            {
                if (vis[i] == 0) InitialDfs(i, vis, adjList, st);
            }
            // List<List<int>> adjT = new List<List<int>>();
            // for(int i=0; i<V; i++) adjT[i] = new List<int>();
            // for(int i=0; i<V; i++){
            //     vis[i] = 0;
            //     foreach(var a in adj[i]){
            //         adjT[a].Add(i);
            //     }
            // }
            for (int i = 0; i < V; i++) vis[i] = 0;
            int count = 0;
            while (st.Count > 0)
            {
                var curr = st.Pop();
                if (vis[curr] == 0)
                {
                    LaterDfs(curr, vis, adjTList);
                    count++;
                }
            }
            return count;
        }
        catch(Exception ex)
        {
            Console.WriteLine($"Exception: {ex.Message}, stack: {ex.StackTrace}");
            return 0;
        }
        
    }

    public static int LengthOfLIS(int[] nums)
    {
        var dp = new List<List<int>>();
        for (int i = 0; i <= nums.Length + 1; i++)
        {
            var list = new List<int>();
            for (int j = 0; j <= nums.Length + 1; j++) list.Add(0);
            dp.Add(list);
        }
        for (int ind = nums.Length; ind > 0; ind--)
        {
            for (int prev_ind = 0; prev_ind <= nums.Length + 1; prev_ind++)
            {
                dp[ind][prev_ind] = dp[ind + 1][prev_ind];
                if (prev_ind == 0 || nums[ind - 1] > nums[prev_ind - 1])
                {
                    dp[ind][prev_ind] = Math.Max(dp[ind][prev_ind], 1 + dp[ind + 1][ind]);
                }
            }
        }
        return dp[1][0];
    }
    class Pair(int a, int b)
    {
        public int dist = a;
        public int node = b;
    }

    class Tripplet(int a, int b, int c)
    {
        public int price = a;
        public int node = b;
        public int stops = c;
    }

    public class Job(int id, int dead, int profit)
    {
        public int id = id;
        public int dead = dead;
        public int profit = profit;
    }

    public class Data(int a, int b, int c)
    {
        public int dist = a;
        public int row = b;
        public int col = c;
    }
}


