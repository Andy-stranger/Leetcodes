using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeetCodes
{
    public class Twitter
    {
        public PriorityQueue<Dictionary<int, int>, int> tweets;
        public Dictionary<int, List<int>> followMap;
        public int post;

        public Twitter()
        {
            tweets = new PriorityQueue<Dictionary<int, int>, int>();
            followMap = new Dictionary<int, List<int>>();
            post = 0;
        }

        public void PostTweet(int userId, int tweetId)
        {
            post++;
            var tweet = new Dictionary<int, int>
            {
                { userId, tweetId }
            };
            tweets.Enqueue(tweet, -post);
        }

        public IList<int> GetNewsFeed(int userId)
        {
            IList<int> result = new List<int>();
            int count = 10;
            Stack<Dictionary<Dictionary<int, int>, int>> stack = new();
            List<int> applicableUserids = new List<int> { userId };
            if (followMap.ContainsKey(userId)) applicableUserids.AddRange(followMap[userId]);
            while (count > 0)
            {
                if (tweets.Count == 0) break;
                stack.Push(new Dictionary<Dictionary<int, int>, int>() { { tweets.Dequeue(), post } });
                post--;
                if (applicableUserids.Contains(stack.Peek().Keys.First().Keys.First()))
                {
                    result.Add(stack.Peek().Keys.First().Values.First());
                    count--;
                }
            }
            while (stack.Count > 0)
            {
                tweets.Enqueue(stack.Pop().Keys.First(), -post);
                post++;
            }
            return result;
        }

        public void Follow(int followerId, int followeeId)
        {
            if (!followMap.ContainsKey(followerId)) followMap.Add(followerId, new List<int>());
            followMap[followerId].Add(followeeId);
        }

        public void Unfollow(int followerId, int followeeId)
        {
            if (followMap.ContainsKey(followerId)) followMap[followerId].Remove(followeeId);
        }
    }
}


