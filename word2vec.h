#include "word.h"
#include "word_vector.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <list>
#include <string>
#include <queue>
#include <unordered_map>
#include <stack>
#include <utility>
#include <algorithm>
#include <memory>
#include <random>
#include <fstream>
#include <sstream>
#include <cassert>


struct TModelConfig {
    size_t LayerSize = 200;
    int Window = 5;
    size_t BatchSize = 800;
    float Sample = 0.0f;
    size_t MinCount = 5;
    float Alpha = 0.025f;
    float MinAlpha = 0.0001f;
    size_t TrainSentenceMaxSize = 1000;
    float TrainSentenceMaxExp = 6.0f;
};

std::ostream& operator<<(std::ostream& os, const TModelConfig& config) {
    os << config.LayerSize << ' '
       << config.Window << ' '
       << config.BatchSize << ' '
       << config.Sample << ' '
       << config.MinCount << ' '
       << config.Alpha << ' '
       << config.MinAlpha << ' '
       << config.TrainSentenceMaxSize << ' '
       << config.TrainSentenceMaxExp;
    return os;
}

std::istream& operator>>(std::istream& is, TModelConfig& config) {
    is >> config.LayerSize
       >> config.Window
       >> config.BatchSize
       >> config.Sample
       >> config.MinCount
       >> config.Alpha
       >> config.MinAlpha
       >> config.TrainSentenceMaxSize
       >> config.TrainSentenceMaxExp;
    return is;
}

class TModel {
private:
	std::vector<TVector> Syn0, Syn1;
	std::vector<TVector> Syn0Norm;

    std::vector<TWordPtr> Words;
    std::unordered_map<std::string, TWord*> WordsIndex;

    std::vector<float> Table;

    TModelConfig Config;

    using Sample = std::vector<TSentence *>;
	using SampleUniqPtr = std::unique_ptr<Sample>;
    using TSamples = std::list<SampleUniqPtr>;

public:
    explicit TModel(const TModelConfig& config)
        : Config(config)
    {}

	void Init(const std::vector<TSentencePtr>& sentences) {
        std::cout << "Start init" << std::endl;

        InitWords(sentences);
        std::cout << "Collected " << WordsIndex.size() << " distinct words" << std::endl;

        InitSyn0Syn1();
        InitTable();

        size_t maxDepth = BuildHuffmanTree();
        std::cout << "Builded huffman tree with max node depth " << maxDepth << std::endl;
    }

	void Train(std::vector<TSentencePtr>& sentences) {
		const size_t totalWords = CalcTotalWords();
		TSamples samples = GenerateSamples(sentences, totalWords);

        int currentWords = 0;
		float alpha0 = Config.Alpha;
        float minAlpha = Config.MinAlpha;
    	while (!samples.empty()) {
			SampleUniqPtr sample = std::move(samples.front());
            samples.pop_front();

			float alpha = std::max(minAlpha, float(alpha0 * (1.0f - 1.0f * currentWords / totalWords)));
			for (auto sentence: *sample) {
				currentWords += TrainSentence(*sentence, alpha);
			}
            std::cout << std::fixed
                      << "alpha: " << std::setprecision(4) << alpha
                      << ", progress: " << std::setprecision(2) << currentWords * 100.0 / totalWords << std::endl;
		}

		Syn0Norm = Syn0;
		for (auto& v: Syn0Norm) {
            Unit(v);
        }
	}

	void Save(const std::string& fileName) const {
		std::ofstream out(fileName, std::ofstream::out);
		out << Config << std::endl;
        out << Syn0.size() << ' ' << Syn0[0].size() << std::endl;

		for (const auto& word: Words) {
			out << word->Text;
			for (const auto& s0: Syn0[word->Index]) {
                out << ' ' << s0;
            }
			out << std::endl;
		}
	}

	static TModel Load(const std::string& fileName) {
		std::ifstream in(fileName);
		std::string line;
        if (!std::getline(in, line)) {
            throw std::runtime_error("Can't read from file");
        }
        std::istringstream iss1(line);
        TModelConfig config;
        iss1 >> config;

        if (!std::getline(in, line)) {
            throw std::runtime_error("Can't read from file");
        }
		std::istringstream iss2(line);
		size_t wordsCount = 0;
        size_t layerSize = 0;
		iss2 >> wordsCount >> layerSize;

        TModel model(config);
		model.Syn0.resize(wordsCount);
		for (size_t i = 0; i < wordsCount; ++i) {
            if (!std::getline(in, line)) {
                throw std::runtime_error("Can't read from file");
            }

			std::istringstream iss(line);
			std::string text;
			iss >> text;

            model.Words.push_back(std::make_shared<TWord>(i, text, 0));
			model.WordsIndex.emplace(text, model.Words.back().get());
			model.Syn0[i].resize(layerSize);
			for(size_t j = 0; j < layerSize; ++j) {
				iss >> model.Syn0[i][j];
			}
		}

		assert(model.Config.LayerSize == layerSize);

		model.Syn0Norm = model.Syn0;
		for (auto& v: model.Syn0Norm) {
            Unit(v);
        }
        return model;
	}

    std::vector<std::pair<std::string, float>> GetMostSimilar(const std::string& word, size_t topN) const {
        return GetMostSimilar({word}, {}, topN);
    }

	std::vector<std::pair<std::string,float>> GetMostSimilar(const std::vector<std::string>& positiveWords,
                                                             const std::vector<std::string>& negativeWords,
                                                             size_t topN) const {
		if ((positiveWords.empty() && negativeWords.empty()) || Syn0Norm.empty()) {
            return {};
        }

		TVector mean(Config.LayerSize);
		std::vector<int> allWords;
		auto addWord = [&mean, &allWords, this](const std::string& word, float weight) {
			const auto it = WordsIndex.find(word);
			if (it == WordsIndex.end()) {
                return;
            }

			const size_t wordIndex = it->second->Index;
			Saxpy(mean, weight, Syn0Norm[wordIndex]);
			allWords.push_back(wordIndex);
		};

		for (const auto& word: positiveWords) {
            addWord(word, 1.0);
        }
		for (const auto& word: negativeWords) {
            addWord(word, -1.0);
        }

		Unit(mean);

		TVector dists;
		std::vector<size_t> indexes;
		size_t index=0;
		dists.reserve(Syn0Norm.size());
		indexes.reserve(Syn0Norm.size());
		for (const auto& sI: Syn0Norm) {
			dists.push_back(Dot(sI, mean));
			indexes.push_back(index);
            ++index;
		}

		auto comp = [&dists](size_t idx1, size_t idx2) { return dists[idx1] > dists[idx2]; };
	    const size_t shift = std::min(topN + allWords.size(), indexes.size() - 1);
        auto last = indexes.begin() + shift;
		std::make_heap(indexes.begin(), last + 1, comp);
		std::pop_heap(indexes.begin(), last + 1, comp);
		for (auto it = last + 1; it != indexes.end(); ++it) {
			if (!comp(*it, indexes[0])) {
                continue;
            }
		    *last = *it;
		    std::pop_heap(indexes.begin(), last + 1, comp);
		}

		std::sort_heap(indexes.begin(), last, comp);

		std::vector<std::pair<std::string,float>> results;
		for (size_t i = 0; i < shift; ++i) {
			if (std::find(allWords.begin(), allWords.end(), indexes[i]) != allWords.end())
				continue;
			results.push_back(std::make_pair(Words[indexes[i]]->Text, dists[indexes[i]]));
			if (results.size() >= topN) {
                break;
            }
		}

		return results;
	}

private:
	size_t TrainSentence(TSentence& sentence, float alpha) {
		size_t count = 0;
		const int len = sentence.Words.size();
        std::random_device rd;
        std::mt19937 gen(rd());
		std::uniform_int_distribution<int> rng;
        int reducedWindow = rng(gen) % Config.Window;
		for (int i = 0; i < len; ++i) {
			const TWord& current = *sentence.Words[i];
			const int wordsSize = std::min(len, i + Config.Window + 1 - reducedWindow);
			for (int j = std::max(0, i - Config.Window + reducedWindow); j < wordsSize; ++j) {
				const TWord* word = sentence.Words[j];
				if (j == i || word->Codes.empty()) {
					continue;
                }
				auto& l1 = Syn0[word->Index];

				TVector work(Config.LayerSize);
				for (size_t b = 0; b < current.Codes.size(); ++b) {
					auto& l2 = Syn1[current.Points[b]];

					const float e = Dot(l1, l2);
					if (std::abs(e) >= Config.TrainSentenceMaxExp) {
						continue;
                    }

					const size_t fi = (e + Config.TrainSentenceMaxExp)
                                        * (Config.TrainSentenceMaxSize / Config.TrainSentenceMaxExp / 2.0);

					const float f = Table[fi];
					const float g = (1 - current.Codes[b] - f) * alpha;

					Saxpy(work, g, l2);
					Saxpy(l2, g, l1);

				}
                Add(l1, work);
			}
			++count;
		}
		return count;
	}

    void InitWords(const std::vector<TSentencePtr>& sentences) {
        std::unordered_map<std::string, size_t> vocab;
		for (const auto& sentence: sentences) {
			std::string lastToken;
			for (const auto& token: sentence->Tokens) {
				vocab[token] += 1;
			}
		}

		Words.reserve(vocab.size());

		for (const auto& wordCountPair : vocab) {
			size_t count = wordCountPair.second;
			if (count <= Config.MinCount) {
                continue;
            }

            const auto& word = wordCountPair.first;
            Words.push_back(std::make_shared<TWord>(0, word, count));
            WordsIndex.emplace(word, Words.back().get());
		}

        auto comp = [](const TWordPtr& w1, const TWordPtr& w2) { return w1->Count > w2->Count; };
		std::sort(Words.begin(), Words.end(), comp);
        size_t index = 0;
		for (auto& word: Words) {
            word->Index = index;
            ++index;
        }
    }

    void InitSyn0Syn1() {
		Syn0.resize(Words.size());
		Syn1.resize(Words.size());
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> rng(0.0f, 1.0f);
		for (auto& s0 : Syn0) {
			s0.resize(Config.LayerSize);
			for (auto& s0I : s0) {
                s0I = (rng(gen) - 0.5f) / Config.LayerSize;
            }
		}
		for (auto& s1 : Syn1) {
            s1.resize(Config.LayerSize);
        }
    }

    void InitTable() {
		Table.resize(Config.TrainSentenceMaxSize);
		for (size_t i = 0; i < Config.TrainSentenceMaxSize; ++i) {
            float e = exp((i / float(Config.TrainSentenceMaxSize) * 2.0 - 1.0) * Config.TrainSentenceMaxExp);
            Table[i] = e / (e + 1);
        }
    }

    size_t BuildHuffmanTree() {
		auto comp = [](const TWord* w1, const TWord* w2) { return w1->Count > w2->Count; };
        std::priority_queue<TWord*, std::vector<TWord*>, decltype(comp)> heap(comp);
        for (const auto& word : Words) {
            heap.push(word.get());
        }

		std::vector<TWordPtr> heapObjVault;
		for (size_t i = 0; i + 1 < Words.size(); ++i) {
            auto min1 = heap.top();
            heap.pop();
			auto min2 = heap.top();
            heap.pop();

            heapObjVault.push_back(std::make_shared<TWord>(i + Words.size(), "", min1->Count + min2->Count, min1, min2));
			heap.push(heapObjVault.back().get());
		}

		size_t maxDepth = 0;
        std::stack<std::pair<TWord*, size_t>> stack;
        stack.emplace(heap.top(), 0);
        while (!stack.empty()) {
			auto cur = stack.top();
			TWord* word = cur.first;
            const size_t curDepth = cur.second;
            stack.pop();

            maxDepth = std::max(maxDepth, curDepth);

            if (!word->LeftPtr) {
                continue;
            }

            word->LeftPtr->Codes = word->Codes;
            word->LeftPtr->Codes.push_back(0);
            word->RightPtr->Codes = word->Codes;
            word->RightPtr->Codes.push_back(1);

            word->Points.push_back(word->Index - Words.size());
            word->LeftPtr->Points = word->Points;
            word->RightPtr->Points = word->Points;

            stack.emplace(word->LeftPtr, curDepth + 1);
            stack.emplace(word->RightPtr, curDepth + 1);
		}
        return maxDepth;
    }

    TSamples GenerateSamples(std::vector<TSentencePtr>& sentences, size_t totalWords) const {
        TSamples samples;
        std::random_device rd;
        std::mt19937 gen(rd());
		std::uniform_real_distribution<float> rng(0.0f, 1.0f);

		SampleUniqPtr sample = std::make_unique<Sample>();
		for (auto& sentence: sentences) {
			if (sentence->Tokens.empty()) {
				continue;
            }
			sentence->Words.reserve(sentence->Tokens.size());
			for (const auto& token : sentence->Tokens) {
				auto iter = WordsIndex.find(token);
				if (iter == WordsIndex.end()) {
                    continue;
                }
				TWord* word = iter->second;
				if (Config.Sample > 0) {
					const float ran = (sqrt(word->Count / (Config.Sample * totalWords)) + 1)
                                        * (Config.Sample * totalWords) / word->Count;
					if (ran < rng(gen)) {
                        continue;
                    }
				}
				sentence->Words.push_back(word);
			}

			sample->push_back(sentence.get());
			if (sample->size() == Config.BatchSize) {
				samples.push_back(std::move(sample));
				sample = std::make_unique<Sample>();
			}
		}

		if (!sample->empty()) {
			samples.push_back(std::move(sample));
        }

        return samples;
    }


    size_t CalcTotalWords() const {
        size_t totalWords = 0;
        for (const auto& word : Words) {
            totalWords += word->Count;
        }
        return totalWords;
    }
};

