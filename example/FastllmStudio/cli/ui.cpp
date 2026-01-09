//
// Created by huangyuyang on 5/17/24.
//

#include "ui.h"

#ifdef _WIN32
#include <conio.h>
#endif

namespace fastllmui {
    enum class Key {
        Enter,
        Up,
        Down,
        Other,
    };

    static Key readKey() {
#ifdef _WIN32
        const int ch = _getch();
        if (ch == '\r' || ch == '\n') {
            return Key::Enter;
        }
        // Windows console: arrow keys are returned as 0/224 prefix + scan code
        if (ch == 0 || ch == 224) {
            const int scan = _getch();
            if (scan == 72) return Key::Up;
            if (scan == 80) return Key::Down;
            return Key::Other;
        }
        return Key::Other;
#else
        static char ch;
        int ret = system("stty -icanon -echo");
        ret = scanf("%c", &ch);
        ret = system("stty icanon echo");
        (void)ret;

        if (ch == '\r' || ch == '\n') {
            return Key::Enter;
        }

        // ANSI escape sequences: ESC [ A/B
        static std::string buffer;
        buffer.push_back(ch);
        const std::string upString = {27, 91, 65};
        const std::string downString = {27, 91, 66};
        if (buffer.size() > 3) {
            buffer.erase(0, buffer.size() - 3);
        }
        if (buffer.size() == 3 && buffer == upString) {
            buffer.clear();
            return Key::Up;
        }
        if (buffer.size() == 3 && buffer == downString) {
            buffer.clear();
            return Key::Down;
        }
        return Key::Other;
#endif
    }

    void PrintNormalLine(const std::string &line) {
        printf("%s", line.c_str());
    }

    void PrintHighlightLine(const std::string &line) {
        printf("\e[1;31;40m %s \e[0m",  line.c_str());
    }

    void HideCursor() {
        printf("\033[?25l");
    }

    void ShowCursor() {
        printf("\033[?25h");
    }

    void ClearScreen() {
        printf("\033c");
    }

    void CursorUp() {
        printf("\033[F");
    }

    void CursorDown() {
        printf("\033[B");
    }

    void CursorClearLine() {
        printf("\033[1G");
        printf("\033[K");
    }

    int Menu::Show() {
        for (int i = 0; i < items.size(); i++) {
            if (i == curIndex) {
                PrintHighlightLine(items[i]);
                printf("\n");
            } else {
                PrintNormalLine(items[i]);
                printf("\n");
            }
        }

        for (int i = curIndex; i < items.size(); i++) {
            printf("\033[F");
        }

        while (true) {
            const Key key = readKey();
            if (key == Key::Enter) {
                return curIndex;
            }
            if (key == Key::Down) {
                if (curIndex + 1 < (int)items.size()) {
                    CursorClearLine();
                    PrintNormalLine(items[curIndex++]);
                    CursorDown();
                    CursorClearLine();
                    PrintHighlightLine(items[curIndex]);
                }
            }
            if (key == Key::Up) {
                if (curIndex - 1 >= 0) {
                    CursorClearLine();
                    PrintNormalLine(items[curIndex--]);
                    CursorUp();
                    CursorClearLine();
                    PrintHighlightLine(items[curIndex]);
                }
            }
        }
    }
} // namespace fastllmui
