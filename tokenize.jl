include("./sentencepiece.jl")


function main()
    vocab = SentencePiece.load_vocab()
    for line in readlines()
        tokens = SentencePiece.pretokenize(line)
        subwords = Vector{String}()
        for token in tokens
            fw_score = last(SentencePiece.forward(token, vocab))
            bw_score = SentencePiece.backward(token, vocab)[1]

            append!(subwords, SentencePiece.viterbi_segment(token, vocab))
        end
        println(join(subwords, " "))
    end
end

main()
