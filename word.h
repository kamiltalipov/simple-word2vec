#pragma once

#include <vector>
#include <string>
#include <memory>

struct TWord {
    size_t Index = 0;
    std::string Text;
    size_t Count = 0;
    TWord* LeftPtr = nullptr;
    TWord* RightPtr = nullptr;

    std::vector<uint8_t> Codes;
    std::vector<uint32_t> Points;

    TWord(size_t index, const std::string& text, size_t count,
            TWord* leftPtr = nullptr, TWord* rightPtr = nullptr)
        : Index(index)
        , Text(text)
        , Count(count)
        , LeftPtr(leftPtr)
        , RightPtr(rightPtr)
    {}
    TWord(const TWord&) = delete;
    const TWord& operator = (const TWord&) = delete;
};

using TWordPtr = std::shared_ptr<TWord>;

struct TSentence {
    std::vector<TWord*> Words;
    std::vector<std::string> Tokens;
};

using TSentencePtr = std::shared_ptr<TSentence>;

