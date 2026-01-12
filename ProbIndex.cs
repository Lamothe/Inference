namespace Llama;

public struct ProbIndex : IComparable<ProbIndex>
{
    public float prob;
    public int index;
    public int CompareTo(ProbIndex other) => other.prob.CompareTo(this.prob); // Descending
}