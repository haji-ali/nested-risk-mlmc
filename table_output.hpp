#ifndef __TABLE_OUTPUT_HPP__
#define __TABLE_OUTPUT_HPP__

template<typename charT, typename traits = std::char_traits<charT> >
class center_helper {
    std::basic_string<charT, traits> str_;
public:
    center_helper(std::basic_string<charT, traits> str) : str_(str) {}
    template<typename a, typename b>
    friend std::basic_ostream<a, b>& operator<<(std::basic_ostream<a, b>& s, const center_helper<a, b>& c);
};

template<typename charT, typename traits = std::char_traits<charT> >
center_helper<charT, traits> centered(std::basic_string<charT, traits> str) {
    return center_helper<charT, traits>(str);
}

// redeclare for std::string directly so we can support anything that implicitly converts to std::string
inline center_helper<std::string::value_type, std::string::traits_type> centered(const std::string& str) {
    return center_helper<std::string::value_type, std::string::traits_type>(str);
}

template<typename charT, typename traits>
std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& s, const center_helper<charT, traits>& c) {
    std::streamsize w = s.width();
    if (w > static_cast<int>(c.str_.length())) {
        std::streamsize left = (w + c.str_.length()) / 2;
        s.width(left);
        s << c.str_;
        s.width(w - left);
        s << "";
    } else {
        s << c.str_;
    }
    return s;
}

struct table_t {
    struct field_t {
        field_t(int _min_len,
                std::function<void (std::stringstream&, uint32_t)> _value)
            : min_len(_min_len), value(_value) {
        }

        size_t min_len;
        std::function<void (std::stringstream&, uint32_t)> value;
    };

    table_t(uint32_t _rows, bool na=false): non_ascii(na), rows(_rows) {}
    bool non_ascii;   // If true, use modifiers for better output
    uint32_t rows;
    std::vector<field_t> fields;

    void add(const std::string& _header,
             int _min_len,
             std::function<void (std::stringstream&, uint32_t)> _value){
        fields.push_back(
            field_t({_min_len,
                     [_header, _value] (std::stringstream& o, uint32_t i) {
                         if (i==0) o << centered(_header);
                         else _value(o, i-1);
                     }}));
    }
};

inline std::stringstream& operator<<(std::stringstream& s, table_t& tbl) {
    const char * underline = "\033[4m";
    const char * underline_bold = "\033[4;1m";
    const char * normal = "\033[0m";

    using namespace std;
    uint32_t rows = tbl.rows + 1; // + header
    for(auto& f : tbl.fields) {
        for (uint32_t i=0;i<rows;i++){
            stringstream tmp;
            tmp.copyfmt(s);
            tmp << setw(0);
            f.value(tmp, i);
            f.min_len = std::max(tmp.str().size(), f.min_len);
        }
    }

    for (uint32_t i=0;i<rows;i++){
        bool do_line = (i==0 || i == rows-1);
        if (do_line && tbl.non_ascii)
            s << underline;
        s << "| ";
        for(const auto& f : tbl.fields) {
            if (i==0 && tbl.non_ascii)
                s << underline_bold;
            
            s << setw(f.min_len);
            f.value(s, i);
            if (i==0 && tbl.non_ascii)
                s << normal << underline;
            s << " | ";
        }
        if (do_line && tbl.non_ascii)
            s << normal;
        s << std::endl;

        // If header or last row, add underline
        if (do_line && not tbl.non_ascii){
            s << "+" << setfill('-');
            for(const auto& f : tbl.fields) {
                s << setw(3+f.min_len) << "+" ;
            }
            s << setfill(' ') << std::endl;
        }
    }
    return s;
}

#endif // __TABLE_OUTPUT_HPP__
