module SentencePiece

export load_vocab, tokenize

import LogExpFunctions


function load_vocab(path::String)
    vocab = Dict{String,Float32}()

    open(path) do file
        for line in eachline(file)
            token, logprob = split(line, "\t")
            vocab[token] = parse(Float32, logprob)
        end
    end
    return vocab
end


function pretokenize(str::String)
    str = "▁" * replace(str, " " => "▁")
    tokens = Vector{String}()
    token_start = 1
    prev_index = 1
    current_index = nextind(str, 1)
    while current_index <= sizeof(str)
        if !isletter(str[current_index]) && !isdigit(str[current_index]) && str[prev_index] != '▁'
            push!(tokens, str[token_start:prev_index])
            token_start = current_index
        end
        prev_index = current_index
        if nextind(str, current_index) > sizeof(str)
            break
        end
        current_index = nextind(str, current_index)
    end
    push!(tokens, str[token_start:current_index])
    return tokens
end


function viterbi_segment(str::String, vocab::Dict{String, Float32})
    points = collect(eachindex(str))

    costs = fill(0., length(points) + 1)
    prev = fill(0, length(points))

    # First, dynamic programming
    for (i, point) in enumerate(points)
        best_score = -Inf
        best_index = 0
        for (j, start_point) in enumerate(points[1:i])
            subword_candidate = str[start_point:point]
            if haskey(vocab, subword_candidate)
                new_cost = costs[j] + vocab[subword_candidate]
                if new_cost > best_score
                   best_index = j
                   best_score = new_cost
                end
            end
        end
        if best_index == 0
            costs[i + 1] = -1000
            prev[i] = i
        else
            costs[i + 1] = best_score
            prev[i] = best_index
        end
    end

    # Second, reconstrct the best options
    subwords = Vector{String}()
    idx = length(prev)
    while idx >= 1
        new_idx = prev[idx]
        if new_idx == 0
            break
        end
        push!(subwords, str[points[new_idx]:points[idx]])
        idx = new_idx - 1
    end
    return reverse(subwords)
end


function forward(str::String, vocab::Dict{String, Float32})
    points = collect(eachindex(str))

    costs = fill(0., length(points) + 1)

    for (i, point) in enumerate(points)
        scores = Vector{Float32}()
        for (j, start_point) in enumerate(points[1:i])
            subword_candidate = str[start_point:point]
            if haskey(vocab, subword_candidate)
                new_cost = costs[j] + vocab[subword_candidate]
                push!(scores, new_cost)
            end
        end
        costs[i + 1] = LogExpFunctions.logsumexp(scores)
    end

    return costs[2:length(costs)]
end


function backward(str::String, vocab::Dict{String, Float32})
    points = reverse(collect(eachindex(str)))

    costs = fill(0., length(points) + 1)

    for (i, point) in enumerate(points)
        scores = Vector{Float32}()
        for (j, end_point) in enumerate(points[1:i])
            subword_candidate = str[point:end_point]
            if haskey(vocab, subword_candidate)
                new_cost = costs[j] + vocab[subword_candidate]
                push!(scores, new_cost)
            end
        end
        costs[i + 1] = LogExpFunctions.logsumexp(scores)
    end

    return reverse(costs[2:length(costs)])
end


function token_expected_counts(
        token::String,
        token_count::Int,
        max_subword_len::Int,
        vocab::Dict{String, Float32},
        expected_count_table::Dict{String, Float32})
    forward_probs = forward(token, vocab)
    backward_probs = backward(token, vocab)

    points = collect(eachindex(str))
    for subword_len in range(max_subword_len)
        for start in range(length(points) - max_subword_len)
            prefix_score = forward_probs[start]
            suffix_score = backward_probs[start + subword_len]
            subword = token[points[start]:points[start + subword_len]]
        end
    end
end


function tokenize(sentence::String, vocab::Dict{String, Float32})
    tokens = pretokenize(sentence)
    subwords = Vector{String}()
    for token in tokens
        fw_score = last(SentencePiece.forward(token, vocab))
        bw_score = SentencePiece.backward(token, vocab)[1]

        append!(subwords, SentencePiece.viterbi_segment(token, vocab))
    end
    return subwords
end


function get_token_counts(path::String)
    token_counts = Dict{String,Float32}()

    open(path) do file
        for line in eachline(file)
            for tok in pretokenize(line)
                if !haskey(token_counts, tok)
                    token_counts[token] = 0
                end
                token_counts[token] += 1
            end
        end
    end
    return token_counts
end

end
