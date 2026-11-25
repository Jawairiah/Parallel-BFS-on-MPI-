#include <mpi.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int u, v; } Edge;
static int rank, ranks;

static void fail(const char *message)
{
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(EXIT_FAILURE);
}

static void *array(size_t count, size_t width)
{
    void *p = calloc(count ? count : 1, width);
    if (!p) fail("Out of memory");
    return p;
}

static int owner(int vertex, int block)
{
    return vertex / block;
}

/* SNAP roadNet-CA uses integer IDs. Missing IDs are represented as isolated vertices. */
static void load(const char *path, int *vertices, int **offset, int **neighbors)
{
    FILE *input = fopen(path, "r");
    if (!input) fail("Cannot open graph file");
    char line[256];
    int maximum = -1, edges = 0;
    while (fgets(line, sizeof line, input)) {
        if (line[0] == '#') continue;
        Edge edge;
        if (sscanf(line, "%d%d", &edge.u, &edge.v) != 2 ||
            edge.u < 0 || edge.v < 0 || edge.u >= INT_MAX - 1 ||
            edge.v >= INT_MAX - 1 || edges == INT_MAX / 2)
            fail("Invalid edge list or graph too large");
        if (edge.u > maximum) maximum = edge.u;
        if (edge.v > maximum) maximum = edge.v;
        ++edges;
    }
    if (ferror(input) || edges == 0) fail("Cannot read nonempty graph");
    *vertices = maximum + 1;
    *offset = array((size_t)*vertices + 1, sizeof(int));
    *neighbors = array(2u * edges, sizeof(int));
    Edge *pairs = array(edges, sizeof(Edge));
    rewind(input);
    int index = 0;
    while (fgets(line, sizeof line, input)) {
        if (line[0] == '#') continue;
        if (index >= edges || sscanf(line, "%d%d", &pairs[index].u, &pairs[index].v) != 2 ||
            pairs[index].u < 0 || pairs[index].u > maximum ||
            pairs[index].v < 0 || pairs[index].v > maximum)
            fail("Graph changed while reading");
        ++(*offset)[pairs[index].u + 1];
        ++(*offset)[pairs[index].v + 1];
        ++index;
    }
    if (ferror(input) || index != edges) fail("Could not reread all edges");
    int *cursor = array(*vertices, sizeof(int));
    for (int v = 0; v < *vertices; ++v) {
        (*offset)[v + 1] += (*offset)[v];
        cursor[v] = (*offset)[v];
    }
    for (int i = 0; i < edges; ++i) {
        Edge edge = pairs[i];
        (*neighbors)[cursor[edge.u]++] = edge.v;
        (*neighbors)[cursor[edge.v]++] = edge.u;
    }
    free(cursor);
    free(pairs);
    fclose(input);
}

static void reset(int *dist, int n)
{
    for (int i = 0; i < n; ++i) dist[i] = -1;
}

static double sequential(int n, const int *offset, const int *neighbors,
                         int source, int *dist)
{
    int *queue = array(n, sizeof(int));
    reset(dist, n);
    double start = MPI_Wtime();
    int head = 0, tail = 0;
    queue[tail++] = source;
    dist[source] = 0;
    while (head < tail) {
        int vertex = queue[head++];
        for (int i = offset[vertex]; i < offset[vertex + 1]; ++i) {
            int neighbor = neighbors[i];
            if (dist[neighbor] != -1) continue;
            dist[neighbor] = dist[vertex] + 1;
            queue[tail++] = neighbor;
        }
    }
    double elapsed = MPI_Wtime() - start;
    free(queue);
    return elapsed;
}

static int prefix(const int *counts, int *positions)
{
    int total = 0;
    for (int r = 0; r < ranks; ++r) {
        positions[r] = total;
        if (counts[r] > INT_MAX - total) fail("MPI buffer exceeds integer counts");
        total += counts[r];
    }
    return total;
}

static int vertex_id(const char *text)
{
    char *end;
    long value = strtol(text, &end, 10);
    if (end == text || *end || value < 0 || value >= INT_MAX)
        fail("Vertex ID must be a nonnegative integer");
    return (int)value;
}

static void print_path(int source, int target, const int *offset,
                       const int *neighbors, const int *distance)
{
    int length = distance[target];
    if (length < 0) {
        printf("No path from %d to %d\n", source, target);
        return;
    }
    int *path = array((size_t)length + 1, sizeof(int));
    path[length] = target;
    for (int d = length; d > 0; --d) {
        int vertex = path[d];
        int i = offset[vertex];
        while (i < offset[vertex + 1] && distance[neighbors[i]] != d - 1)
            ++i;
        if (i == offset[vertex + 1]) fail("Could not reconstruct shortest path");
        path[d - 1] = neighbors[i];
    }
    printf("Shortest path (%d edges): %d", length, path[0]);
    for (int d = 1; d <= length; ++d)
        printf(" -> %d", path[d]);
    putchar('\n');
    free(path);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc != 3 && argc != 4) fail("Usage: bfs EDGE_LIST SOURCE_VERTEX [TARGET_VERTEX]");
    int source = vertex_id(argv[2]);
    int target = argc == 4 ? vertex_id(argv[3]) : -1;
    int n = 0, *offset = NULL, *neighbors = NULL;
    double setup_start = MPI_Wtime();
    if (rank == 0) load(argv[1], &n, &offset, &neighbors);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (source >= n) fail("Source lies outside the graph");
    if (target >= n) fail("Target lies outside the graph");

    int block = n / ranks + (n % ranks != 0);
    int *starts = array(ranks, sizeof(int));
    int *sizes = array(ranks, sizeof(int));
    int *edge_starts = array(ranks, sizeof(int));
    int *edge_counts = array(ranks, sizeof(int));
    for (int r = 0; r < ranks; ++r) {
        long long start = (long long)r * block;
        starts[r] = start < n ? (int)start : n;
        sizes[r] = n - starts[r] < block ? n - starts[r] : block;
        if (rank == 0) {
            edge_starts[r] = offset[starts[r]];
            edge_counts[r] = offset[starts[r] + sizes[r]] - edge_starts[r];
        }
    }
    int local_n = sizes[rank], first = starts[rank], local_edges;
    MPI_Scatter(edge_counts, 1, MPI_INT, &local_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int *local_offset = array((size_t)local_n + 1, sizeof(int));
    int *local_neighbors = array(local_edges, sizeof(int));
    MPI_Scatterv(offset, sizes, starts, MPI_INT, local_offset, local_n, MPI_INT, 0,
                 MPI_COMM_WORLD);
    MPI_Scatterv(neighbors, edge_counts, edge_starts, MPI_INT, local_neighbors,
                 local_edges, MPI_INT, 0, MPI_COMM_WORLD);
    if (local_n) {
        int base = local_offset[0];
        for (int v = 0; v < local_n; ++v) local_offset[v] -= base;
    }
    local_offset[local_n] = local_edges;
    int *dist = array(local_n, sizeof(int));
    int *frontier = array(local_n, sizeof(int));
    int *next = array(local_n, sizeof(int));
    int *send_counts = array(ranks, sizeof(int));
    int *recv_counts = array(ranks, sizeof(int));
    int *send_pos = array(ranks, sizeof(int));
    int *recv_pos = array(ranks, sizeof(int));
    /* Reserve a fixed segment per destination. Each local vertex is expanded
       at most once, so its outgoing edges fit within these bounds at any level. */
    int *send_limits = array(ranks, sizeof(int));
    for (int i = 0; i < local_edges; ++i) {
        int neighbor = local_neighbors[i];
        if ((unsigned)(neighbor - first) >= (unsigned)local_n)
            ++send_limits[owner(neighbor, block)];
    }
    int *send = array(prefix(send_limits, send_pos), sizeof(int));
    free(send_limits);
    int *receive = NULL;
    int receive_capacity = 0, frontier_size = 0;
    double local_setup = MPI_Wtime() - setup_start, setup;
    MPI_Reduce(&local_setup, &setup, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    reset(dist, local_n);
    if (owner(source, block) == rank) {
        frontier[frontier_size++] = source - first;
        dist[source - first] = 0;
    }
    long long scanned = 0;
    int level = 0;
    int skipped_payload_exchanges = 0;
    double communication = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    while (ranks > 1 || frontier_size > 0) {
        int next_size = 0;
        memset(send_counts, 0, (size_t)ranks * sizeof(int));
        for (int j = 0; j < frontier_size; ++j) {
            int v = frontier[j];
            scanned += local_offset[v + 1] - local_offset[v];
            for (int i = local_offset[v]; i < local_offset[v + 1]; ++i) {
                int neighbor = local_neighbors[i];
                int local_neighbor = neighbor - first;
                if ((unsigned)local_neighbor < (unsigned)local_n) {
                    if (dist[local_neighbor] == -1) {
                        dist[local_neighbor] = level + 1;
                        next[next_size++] = local_neighbor;
                    }
                } else {
                    int destination = owner(neighbor, block);
                    send[send_pos[destination] + send_counts[destination]++] = neighbor;
                }
            }
        }
        if (ranks > 1) {
            int active = next_size != 0;
            for (int r = 0; r < ranks; ++r)
                active |= send_counts[r] != 0;

            /* Negative counts carry the sender's activity flag to every rank.
               Encoding as -(count + 1) also flags a zero-count destination. */
            if (active) {
                for (int r = 0; r < ranks; ++r)
                    send_counts[r] = -send_counts[r] - 1;
            }
            double comm_start = MPI_Wtime();
            MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
            communication += MPI_Wtime() - comm_start;
            int any_active = 0;
            for (int r = 0; r < ranks; ++r) {
                if (recv_counts[r] < 0) {
                    any_active = 1;
                    recv_counts[r] = -(recv_counts[r] + 1);
                }
                if (active)
                    send_counts[r] = -(send_counts[r] + 1);
            }
            if (!any_active) break;

            /* With two ranks, both see the same pair of peer counts in
               opposite order. With more ranks, local counts cannot prove
               that there is no traffic between other pairs of ranks. */
            if (ranks == 2 && send_counts[1 - rank] == 0 && recv_counts[1 - rank] == 0) {
                ++skipped_payload_exchanges;
            } else {
                int required = prefix(recv_counts, recv_pos);
                if (required > receive_capacity) {
                    int *grown = realloc(receive, (size_t)required * sizeof(int));
                    if (!grown) fail("Out of memory growing receive buffer");
                    receive = grown;
                    receive_capacity = required;
                }
                int empty = 0;
                comm_start = MPI_Wtime();
                MPI_Alltoallv(send, send_counts, send_pos, MPI_INT, receive ? receive : &empty,
                              recv_counts, recv_pos, MPI_INT, MPI_COMM_WORLD);
                communication += MPI_Wtime() - comm_start;
                for (int i = 0; i < required; ++i) {
                    int v = receive[i] - first;
                    if (dist[v] != -1) continue;
                    dist[v] = level + 1;
                    next[next_size++] = v;
                }
            }
        }
        int *swap = frontier;
        frontier = next;
        next = swap;
        frontier_size = next_size;
        ++level;
    }
    struct { double seconds; int rank; } local = {MPI_Wtime() - start, rank}, slowest;
    MPI_Allreduce(&local, &slowest, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    double slow_comm = rank == slowest.rank ? communication : 0, total_comm;
    long long total_scanned;
    MPI_Reduce(&slow_comm, &total_comm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&scanned, &total_scanned, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    int *result = rank == 0 ? array(n, sizeof(int)) : NULL;
    MPI_Gatherv(dist, local_n, MPI_INT, result, sizes, starts, MPI_INT, 0, MPI_COMM_WORLD);

    int correct = 1;
    if (rank == 0) {
        int *expected = array(n, sizeof(int));
        double serial = sequential(n, offset, neighbors, source, expected);
        correct = memcmp(expected, result, (size_t)n * sizeof(int)) == 0;
        double parallel = slowest.seconds;
        double speedup = parallel > 0 ? serial / parallel : 0;
        printf("ranks,source,setup_s,serial_s,mpi_s,speedup,efficiency_pct,scanned_adjacencies,million_edges_per_s,communication_pct,skipped_payload_exchanges,verified\n");
        printf("%d,%d,%.9f,%.9f,%.9f,%.6f,%.3f,%lld,%.6f,%.3f,%d,%s\n",
               ranks, source, setup, serial, parallel, speedup, 100 * speedup / ranks,
               total_scanned, parallel > 0 ? total_scanned / parallel / 1e6 : 0,
               parallel > 0 ? 100 * total_comm / parallel : 0,
               skipped_payload_exchanges, correct ? "yes" : "no");
        if (correct && target >= 0)
            print_path(source, target, offset, neighbors, result);
        free(expected);
    }
    MPI_Bcast(&correct, 1, MPI_INT, 0, MPI_COMM_WORLD);
    free(offset); free(neighbors); free(starts); free(sizes);
    free(edge_starts); free(edge_counts); free(local_offset); free(local_neighbors);
    free(dist); free(frontier); free(next); free(send_counts); free(recv_counts);
    free(send_pos); free(recv_pos); free(send); free(receive); free(result);
    MPI_Finalize();
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
