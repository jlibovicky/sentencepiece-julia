using ArgParse
using Logging

include("./sentencepiece.jl")


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "vocab"
            help = "File with vocabulary with probabilities."
            required = true
        "input"
            help = "Input file(s)"
            required = false
    end

    return parse_args(s)
end


function main()
    args = parse_commandline()
    @info("Loading vocabulary with log-probabilities.")
    vocab = SentencePiece.load_vocab(args["vocab"])

    if isnothing(args["input"])
        @info("No input file given, using stdin.")
        input = stdin
    else
        @info("Read input from " * args["input"] * ".")
        input = open(args["input"])
    end

    for line in readlines(input)
        subwords = SentencePiece.tokenize(line, vocab)
        #println(join(subwords, " "))
    end
end

main()
