namespace LeetCodes
{
    public static class StringExtensions
    {
        public static long Hash(this string s, int p, int m)
        {
            long hash_value = 0;
            long p_pow = 1;
            foreach (char c in s)
            {
                hash_value = (hash_value + (c - 'a' + 1) * p_pow) % m;
                p_pow = (p_pow * p) % m;
            }
            return hash_value;
        }
    }
}
