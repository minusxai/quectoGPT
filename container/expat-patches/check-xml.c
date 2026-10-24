#include <expat.h>
#include <assert.h>
#include <string.h>
int main(void) { XML_Parser p=XML_ParserCreate(NULL); assert(p); const char *s="<root><item id=\"1\">ordinary UTF-8 text</item></root>"; for(int i=0;i<3;i++){ assert(XML_Parse(p,s,strlen(s),XML_TRUE)==XML_STATUS_OK); assert(XML_ParserReset(p,NULL)==XML_TRUE); } XML_ParserFree(p); return 0; }
