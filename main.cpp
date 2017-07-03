#include "word2vec.h"
#include "cmdline.h"

#include <iostream>
#include <string>

std::vector<TSentencePtr> ReadSentences(const std::string& fileName, size_t maxSentenceLen) {
    std::vector<TSentencePtr> sentences;

    size_t count = 0;

    TSentencePtr sentence = std::make_shared<TSentence>();
    std::ifstream in(fileName);
    std::string s;
    while (in >> s) {
        ++count;
        sentence->Tokens.push_back(std::move(s));
        if (count == maxSentenceLen) {
            count = 0;
            sentences.push_back(std::move(sentence));
            sentence = std::make_shared<TSentence>();
        }
    }

    if (!sentence->Tokens.empty()) {
        sentences.push_back(std::move(sentence));
    }

    return sentences;
}

cmdline::parser GetCmdParser() {
    cmdline::parser cmdParser;
    cmdParser.add("train", '\0', "Train model");
    cmdParser.add("test", '\0', "Test model");
    cmdParser.add<std::string>("input", 'i', "input file name with text", true, "");
    cmdParser.add<std::string>("output", 'o', "output file name with models", true, "");
    cmdParser.add<size_t>("max_sentence_len", '\0', "", false, 200);
    cmdParser.add<size_t>("layer_size", '\0', "", false, 200);
    cmdParser.add<int>("window", '\0', "", false, 5);
    cmdParser.add<size_t>("batch_size", '\0', "", false, 800);
    cmdParser.add<float>("sample", '\0', "", false, 0.0f);
    cmdParser.add<size_t>("min_count", '\0', "", false, 5);
    cmdParser.add<float>("alpha", '\0', "", false, 0.025f);
    cmdParser.add<float>("min_alpha", '\0', "", false, 0.0001f);
    cmdParser.add<size_t>("train_max_size", '\0', "", false, 1000);
    cmdParser.add<float>("train_max_exp", '\0', "", false, 6.0f);

    return cmdParser;
}

int main(int argc, char *argv[])
{
    auto cmdParser = GetCmdParser();
    cmdParser.parse_check(argc, argv);

    bool train = cmdParser.exist("train");
    bool test = cmdParser.exist("test");
    if (train) {
        auto sentences = ReadSentences(cmdParser.get<std::string>("input"),
                                       cmdParser.get<std::size_t>("max_sentence_len"));
        TModelConfig modelConfig;
        modelConfig.LayerSize = cmdParser.get<size_t>("layer_size");
        modelConfig.Window = cmdParser.get<int>("window");
        modelConfig.BatchSize = cmdParser.get<size_t>("batch_size");
        modelConfig.Sample = cmdParser.get<float>("sample");
        modelConfig.MinCount = cmdParser.get<size_t>("min_count");
        modelConfig.Alpha = cmdParser.get<float>("alpha");
        modelConfig.MinAlpha = cmdParser.get<float>("min_alpha");
        modelConfig.TrainSentenceMaxSize = cmdParser.get<size_t>("train_max_size");
        modelConfig.TrainSentenceMaxExp = cmdParser.get<float>("train_max_exp");

        TModel model(modelConfig);
        model.Init(sentences);
        model.Train(sentences);
        model.Save(cmdParser.get<std::string>("output"));
    }

    if (test) {
        TModel model = TModel::Load(cmdParser.get<std::string>("output"));
        while (true) {
            std::string str;
            std::cout << std::endl << "Input(:q to break):";
            std::cin >> str;
            if (str == ":q") {
                break;
            }
            auto res = model.GetMostSimilar(str, 10);
            size_t idx = 0;
            for (const auto& v : res) {
                std::cout << idx << ' ' << v.first << ' ' << v.second << std::endl;
                ++idx;
            }
        }
    }

    return 0;
}

