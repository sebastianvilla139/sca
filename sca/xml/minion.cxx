// Copyright (c) 2005-2014 Code Synthesis Tools CC
//
// This program was generated by CodeSynthesis XSD, an XML Schema to
// C++ data binding compiler.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA
//
// In addition, as a special exception, Code Synthesis Tools CC gives
// permission to link this program with the Xerces-C++ library (or with
// modified versions of Xerces-C++ that use the same license as Xerces-C++),
// and distribute linked combinations including the two. You must obey
// the GNU General Public License version 2 in all respects for all of
// the code used other than Xerces-C++. If you modify this copy of the
// program, you may extend this exception to your version of the program,
// but you are not obligated to do so. If you do not wish to do so, delete
// this exception statement from your version.
//
// Furthermore, Code Synthesis Tools CC makes a special exception for
// the Free/Libre and Open Source Software (FLOSS) which is described
// in the accompanying FLOSSE file.
//

// Begin prologue.
//
//
// End prologue.

#include <xsd/cxx/pre.hxx>

#include "minion.hxx"

// minion
// 

const minion::name_type& minion::
name () const
{
  return this->name_.get ();
}

minion::name_type& minion::
name ()
{
  return this->name_.get ();
}

void minion::
name (const name_type& x)
{
  this->name_.set (x);
}

void minion::
name (::std::auto_ptr< name_type > x)
{
  this->name_.set (x);
}

const minion::rank_type& minion::
rank () const
{
  return this->rank_.get ();
}

minion::rank_type& minion::
rank ()
{
  return this->rank_.get ();
}

void minion::
rank (const rank_type& x)
{
  this->rank_.set (x);
}

void minion::
rank (::std::auto_ptr< rank_type > x)
{
  this->rank_.set (x);
}

const minion::serial_type& minion::
serial () const
{
  return this->serial_.get ();
}

minion::serial_type& minion::
serial ()
{
  return this->serial_.get ();
}

void minion::
serial (const serial_type& x)
{
  this->serial_.set (x);
}

const minion::loyalty_type& minion::
loyalty () const
{
  return this->loyalty_.get ();
}

minion::loyalty_type& minion::
loyalty ()
{
  return this->loyalty_.get ();
}

void minion::
loyalty (const loyalty_type& x)
{
  this->loyalty_.set (x);
}


#include <xsd/cxx/xml/dom/parsing-source.hxx>

// minion
//

minion::
minion (const name_type& name,
        const rank_type& rank,
        const serial_type& serial,
        const loyalty_type& loyalty)
: ::xml_schema::type (),
  name_ (name, this),
  rank_ (rank, this),
  serial_ (serial, this),
  loyalty_ (loyalty, this)
{
}

minion::
minion (const minion& x,
        ::xml_schema::flags f,
        ::xml_schema::container* c)
: ::xml_schema::type (x, f, c),
  name_ (x.name_, f, this),
  rank_ (x.rank_, f, this),
  serial_ (x.serial_, f, this),
  loyalty_ (x.loyalty_, f, this)
{
}

minion::
minion (const ::xercesc::DOMElement& e,
        ::xml_schema::flags f,
        ::xml_schema::container* c)
: ::xml_schema::type (e, f | ::xml_schema::flags::base, c),
  name_ (this),
  rank_ (this),
  serial_ (this),
  loyalty_ (this)
{
  if ((f & ::xml_schema::flags::base) == 0)
  {
    ::xsd::cxx::xml::dom::parser< char > p (e, true, false, true);
    this->parse (p, f);
  }
}

void minion::
parse (::xsd::cxx::xml::dom::parser< char >& p,
       ::xml_schema::flags f)
{
  for (; p.more_content (); p.next_content (false))
  {
    const ::xercesc::DOMElement& i (p.cur_element ());
    const ::xsd::cxx::xml::qualified_name< char > n (
      ::xsd::cxx::xml::dom::name< char > (i));

    // name
    //
    if (n.name () == "name" && n.namespace_ ().empty ())
    {
      ::std::auto_ptr< name_type > r (
        name_traits::create (i, f, this));

      if (!name_.present ())
      {
        this->name_.set (r);
        continue;
      }
    }

    // rank
    //
    if (n.name () == "rank" && n.namespace_ ().empty ())
    {
      ::std::auto_ptr< rank_type > r (
        rank_traits::create (i, f, this));

      if (!rank_.present ())
      {
        this->rank_.set (r);
        continue;
      }
    }

    // serial
    //
    if (n.name () == "serial" && n.namespace_ ().empty ())
    {
      if (!serial_.present ())
      {
        this->serial_.set (serial_traits::create (i, f, this));
        continue;
      }
    }

    break;
  }

  if (!name_.present ())
  {
    throw ::xsd::cxx::tree::expected_element< char > (
      "name",
      "");
  }

  if (!rank_.present ())
  {
    throw ::xsd::cxx::tree::expected_element< char > (
      "rank",
      "");
  }

  if (!serial_.present ())
  {
    throw ::xsd::cxx::tree::expected_element< char > (
      "serial",
      "");
  }

  while (p.more_attributes ())
  {
    const ::xercesc::DOMAttr& i (p.next_attribute ());
    const ::xsd::cxx::xml::qualified_name< char > n (
      ::xsd::cxx::xml::dom::name< char > (i));

    if (n.name () == "loyalty" && n.namespace_ ().empty ())
    {
      this->loyalty_.set (loyalty_traits::create (i, f, this));
      continue;
    }
  }

  if (!loyalty_.present ())
  {
    throw ::xsd::cxx::tree::expected_attribute< char > (
      "loyalty",
      "");
  }
}

minion* minion::
_clone (::xml_schema::flags f,
        ::xml_schema::container* c) const
{
  return new class minion (*this, f, c);
}

minion& minion::
operator= (const minion& x)
{
  if (this != &x)
  {
    static_cast< ::xml_schema::type& > (*this) = x;
    this->name_ = x.name_;
    this->rank_ = x.rank_;
    this->serial_ = x.serial_;
    this->loyalty_ = x.loyalty_;
  }

  return *this;
}

minion::
~minion ()
{
}

#include <istream>
#include <xsd/cxx/xml/sax/std-input-source.hxx>
#include <xsd/cxx/tree/error-handler.hxx>

::std::auto_ptr< ::minion >
minion_ (const ::std::string& u,
         ::xml_schema::flags f,
         const ::xml_schema::properties& p)
{
  ::xsd::cxx::xml::auto_initializer i (
    (f & ::xml_schema::flags::dont_initialize) == 0,
    (f & ::xml_schema::flags::keep_dom) == 0);

  ::xsd::cxx::tree::error_handler< char > h;

  ::xml_schema::dom::auto_ptr< ::xercesc::DOMDocument > d (
    ::xsd::cxx::xml::dom::parse< char > (
      u, h, p, f));

  h.throw_if_failed< ::xsd::cxx::tree::parsing< char > > ();

  return ::std::auto_ptr< ::minion > (
    ::minion_ (
      d, f | ::xml_schema::flags::own_dom, p));
}

::std::auto_ptr< ::minion >
minion_ (const ::std::string& u,
         ::xml_schema::error_handler& h,
         ::xml_schema::flags f,
         const ::xml_schema::properties& p)
{
  ::xsd::cxx::xml::auto_initializer i (
    (f & ::xml_schema::flags::dont_initialize) == 0,
    (f & ::xml_schema::flags::keep_dom) == 0);

  ::xml_schema::dom::auto_ptr< ::xercesc::DOMDocument > d (
    ::xsd::cxx::xml::dom::parse< char > (
      u, h, p, f));

  if (!d.get ())
    throw ::xsd::cxx::tree::parsing< char > ();

  return ::std::auto_ptr< ::minion > (
    ::minion_ (
      d, f | ::xml_schema::flags::own_dom, p));
}

::std::auto_ptr< ::minion >
minion_ (const ::std::string& u,
         ::xercesc::DOMErrorHandler& h,
         ::xml_schema::flags f,
         const ::xml_schema::properties& p)
{
  ::xml_schema::dom::auto_ptr< ::xercesc::DOMDocument > d (
    ::xsd::cxx::xml::dom::parse< char > (
      u, h, p, f));

  if (!d.get ())
    throw ::xsd::cxx::tree::parsing< char > ();

  return ::std::auto_ptr< ::minion > (
    ::minion_ (
      d, f | ::xml_schema::flags::own_dom, p));
}

::std::auto_ptr< ::minion >
minion_ (::std::istream& is,
         ::xml_schema::flags f,
         const ::xml_schema::properties& p)
{
  ::xsd::cxx::xml::auto_initializer i (
    (f & ::xml_schema::flags::dont_initialize) == 0,
    (f & ::xml_schema::flags::keep_dom) == 0);

  ::xsd::cxx::xml::sax::std_input_source isrc (is);
  return ::minion_ (isrc, f, p);
}

::std::auto_ptr< ::minion >
minion_ (::std::istream& is,
         ::xml_schema::error_handler& h,
         ::xml_schema::flags f,
         const ::xml_schema::properties& p)
{
  ::xsd::cxx::xml::auto_initializer i (
    (f & ::xml_schema::flags::dont_initialize) == 0,
    (f & ::xml_schema::flags::keep_dom) == 0);

  ::xsd::cxx::xml::sax::std_input_source isrc (is);
  return ::minion_ (isrc, h, f, p);
}

::std::auto_ptr< ::minion >
minion_ (::std::istream& is,
         ::xercesc::DOMErrorHandler& h,
         ::xml_schema::flags f,
         const ::xml_schema::properties& p)
{
  ::xsd::cxx::xml::sax::std_input_source isrc (is);
  return ::minion_ (isrc, h, f, p);
}

::std::auto_ptr< ::minion >
minion_ (::std::istream& is,
         const ::std::string& sid,
         ::xml_schema::flags f,
         const ::xml_schema::properties& p)
{
  ::xsd::cxx::xml::auto_initializer i (
    (f & ::xml_schema::flags::dont_initialize) == 0,
    (f & ::xml_schema::flags::keep_dom) == 0);

  ::xsd::cxx::xml::sax::std_input_source isrc (is, sid);
  return ::minion_ (isrc, f, p);
}

::std::auto_ptr< ::minion >
minion_ (::std::istream& is,
         const ::std::string& sid,
         ::xml_schema::error_handler& h,
         ::xml_schema::flags f,
         const ::xml_schema::properties& p)
{
  ::xsd::cxx::xml::auto_initializer i (
    (f & ::xml_schema::flags::dont_initialize) == 0,
    (f & ::xml_schema::flags::keep_dom) == 0);

  ::xsd::cxx::xml::sax::std_input_source isrc (is, sid);
  return ::minion_ (isrc, h, f, p);
}

::std::auto_ptr< ::minion >
minion_ (::std::istream& is,
         const ::std::string& sid,
         ::xercesc::DOMErrorHandler& h,
         ::xml_schema::flags f,
         const ::xml_schema::properties& p)
{
  ::xsd::cxx::xml::sax::std_input_source isrc (is, sid);
  return ::minion_ (isrc, h, f, p);
}

::std::auto_ptr< ::minion >
minion_ (::xercesc::InputSource& i,
         ::xml_schema::flags f,
         const ::xml_schema::properties& p)
{
  ::xsd::cxx::tree::error_handler< char > h;

  ::xml_schema::dom::auto_ptr< ::xercesc::DOMDocument > d (
    ::xsd::cxx::xml::dom::parse< char > (
      i, h, p, f));

  h.throw_if_failed< ::xsd::cxx::tree::parsing< char > > ();

  return ::std::auto_ptr< ::minion > (
    ::minion_ (
      d, f | ::xml_schema::flags::own_dom, p));
}

::std::auto_ptr< ::minion >
minion_ (::xercesc::InputSource& i,
         ::xml_schema::error_handler& h,
         ::xml_schema::flags f,
         const ::xml_schema::properties& p)
{
  ::xml_schema::dom::auto_ptr< ::xercesc::DOMDocument > d (
    ::xsd::cxx::xml::dom::parse< char > (
      i, h, p, f));

  if (!d.get ())
    throw ::xsd::cxx::tree::parsing< char > ();

  return ::std::auto_ptr< ::minion > (
    ::minion_ (
      d, f | ::xml_schema::flags::own_dom, p));
}

::std::auto_ptr< ::minion >
minion_ (::xercesc::InputSource& i,
         ::xercesc::DOMErrorHandler& h,
         ::xml_schema::flags f,
         const ::xml_schema::properties& p)
{
  ::xml_schema::dom::auto_ptr< ::xercesc::DOMDocument > d (
    ::xsd::cxx::xml::dom::parse< char > (
      i, h, p, f));

  if (!d.get ())
    throw ::xsd::cxx::tree::parsing< char > ();

  return ::std::auto_ptr< ::minion > (
    ::minion_ (
      d, f | ::xml_schema::flags::own_dom, p));
}

::std::auto_ptr< ::minion >
minion_ (const ::xercesc::DOMDocument& doc,
         ::xml_schema::flags f,
         const ::xml_schema::properties& p)
{
  if (f & ::xml_schema::flags::keep_dom)
  {
    ::xml_schema::dom::auto_ptr< ::xercesc::DOMDocument > d (
      static_cast< ::xercesc::DOMDocument* > (doc.cloneNode (true)));

    return ::std::auto_ptr< ::minion > (
      ::minion_ (
        d, f | ::xml_schema::flags::own_dom, p));
  }

  const ::xercesc::DOMElement& e (*doc.getDocumentElement ());
  const ::xsd::cxx::xml::qualified_name< char > n (
    ::xsd::cxx::xml::dom::name< char > (e));

  if (n.name () == "minion" &&
      n.namespace_ () == "")
  {
    ::std::auto_ptr< ::minion > r (
      ::xsd::cxx::tree::traits< ::minion, char >::create (
        e, f, 0));
    return r;
  }

  throw ::xsd::cxx::tree::unexpected_element < char > (
    n.name (),
    n.namespace_ (),
    "minion",
    "");
}

::std::auto_ptr< ::minion >
minion_ (::xml_schema::dom::auto_ptr< ::xercesc::DOMDocument > d,
         ::xml_schema::flags f,
         const ::xml_schema::properties&)
{
  ::xml_schema::dom::auto_ptr< ::xercesc::DOMDocument > c (
    ((f & ::xml_schema::flags::keep_dom) &&
     !(f & ::xml_schema::flags::own_dom))
    ? static_cast< ::xercesc::DOMDocument* > (d->cloneNode (true))
    : 0);

  ::xercesc::DOMDocument& doc (c.get () ? *c : *d);
  const ::xercesc::DOMElement& e (*doc.getDocumentElement ());

  const ::xsd::cxx::xml::qualified_name< char > n (
    ::xsd::cxx::xml::dom::name< char > (e));

  if (f & ::xml_schema::flags::keep_dom)
    doc.setUserData (::xml_schema::dom::tree_node_key,
                     (c.get () ? &c : &d),
                     0);

  if (n.name () == "minion" &&
      n.namespace_ () == "")
  {
    ::std::auto_ptr< ::minion > r (
      ::xsd::cxx::tree::traits< ::minion, char >::create (
        e, f, 0));
    return r;
  }

  throw ::xsd::cxx::tree::unexpected_element < char > (
    n.name (),
    n.namespace_ (),
    "minion",
    "");
}

#include <ostream>
#include <xsd/cxx/tree/error-handler.hxx>
#include <xsd/cxx/xml/dom/serialization-source.hxx>

void
minion_ (::std::ostream& o,
         const ::minion& s,
         const ::xml_schema::namespace_infomap& m,
         const ::std::string& e,
         ::xml_schema::flags f)
{
  ::xsd::cxx::xml::auto_initializer i (
    (f & ::xml_schema::flags::dont_initialize) == 0);

  ::xml_schema::dom::auto_ptr< ::xercesc::DOMDocument > d (
    ::minion_ (s, m, f));

  ::xsd::cxx::tree::error_handler< char > h;

  ::xsd::cxx::xml::dom::ostream_format_target t (o);
  if (!::xsd::cxx::xml::dom::serialize (t, *d, e, h, f))
  {
    h.throw_if_failed< ::xsd::cxx::tree::serialization< char > > ();
  }
}

void
minion_ (::std::ostream& o,
         const ::minion& s,
         ::xml_schema::error_handler& h,
         const ::xml_schema::namespace_infomap& m,
         const ::std::string& e,
         ::xml_schema::flags f)
{
  ::xsd::cxx::xml::auto_initializer i (
    (f & ::xml_schema::flags::dont_initialize) == 0);

  ::xml_schema::dom::auto_ptr< ::xercesc::DOMDocument > d (
    ::minion_ (s, m, f));
  ::xsd::cxx::xml::dom::ostream_format_target t (o);
  if (!::xsd::cxx::xml::dom::serialize (t, *d, e, h, f))
  {
    throw ::xsd::cxx::tree::serialization< char > ();
  }
}

void
minion_ (::std::ostream& o,
         const ::minion& s,
         ::xercesc::DOMErrorHandler& h,
         const ::xml_schema::namespace_infomap& m,
         const ::std::string& e,
         ::xml_schema::flags f)
{
  ::xml_schema::dom::auto_ptr< ::xercesc::DOMDocument > d (
    ::minion_ (s, m, f));
  ::xsd::cxx::xml::dom::ostream_format_target t (o);
  if (!::xsd::cxx::xml::dom::serialize (t, *d, e, h, f))
  {
    throw ::xsd::cxx::tree::serialization< char > ();
  }
}

void
minion_ (::xercesc::XMLFormatTarget& t,
         const ::minion& s,
         const ::xml_schema::namespace_infomap& m,
         const ::std::string& e,
         ::xml_schema::flags f)
{
  ::xml_schema::dom::auto_ptr< ::xercesc::DOMDocument > d (
    ::minion_ (s, m, f));

  ::xsd::cxx::tree::error_handler< char > h;

  if (!::xsd::cxx::xml::dom::serialize (t, *d, e, h, f))
  {
    h.throw_if_failed< ::xsd::cxx::tree::serialization< char > > ();
  }
}

void
minion_ (::xercesc::XMLFormatTarget& t,
         const ::minion& s,
         ::xml_schema::error_handler& h,
         const ::xml_schema::namespace_infomap& m,
         const ::std::string& e,
         ::xml_schema::flags f)
{
  ::xml_schema::dom::auto_ptr< ::xercesc::DOMDocument > d (
    ::minion_ (s, m, f));
  if (!::xsd::cxx::xml::dom::serialize (t, *d, e, h, f))
  {
    throw ::xsd::cxx::tree::serialization< char > ();
  }
}

void
minion_ (::xercesc::XMLFormatTarget& t,
         const ::minion& s,
         ::xercesc::DOMErrorHandler& h,
         const ::xml_schema::namespace_infomap& m,
         const ::std::string& e,
         ::xml_schema::flags f)
{
  ::xml_schema::dom::auto_ptr< ::xercesc::DOMDocument > d (
    ::minion_ (s, m, f));
  if (!::xsd::cxx::xml::dom::serialize (t, *d, e, h, f))
  {
    throw ::xsd::cxx::tree::serialization< char > ();
  }
}

void
minion_ (::xercesc::DOMDocument& d,
         const ::minion& s,
         ::xml_schema::flags)
{
  ::xercesc::DOMElement& e (*d.getDocumentElement ());
  const ::xsd::cxx::xml::qualified_name< char > n (
    ::xsd::cxx::xml::dom::name< char > (e));

  if (n.name () == "minion" &&
      n.namespace_ () == "")
  {
    e << s;
  }
  else
  {
    throw ::xsd::cxx::tree::unexpected_element < char > (
      n.name (),
      n.namespace_ (),
      "minion",
      "");
  }
}

::xml_schema::dom::auto_ptr< ::xercesc::DOMDocument >
minion_ (const ::minion& s,
         const ::xml_schema::namespace_infomap& m,
         ::xml_schema::flags f)
{
  ::xml_schema::dom::auto_ptr< ::xercesc::DOMDocument > d (
    ::xsd::cxx::xml::dom::serialize< char > (
      "minion",
      "",
      m, f));

  ::minion_ (*d, s, f);
  return d;
}

void
operator<< (::xercesc::DOMElement& e, const minion& i)
{
  e << static_cast< const ::xml_schema::type& > (i);

  // name
  //
  {
    ::xercesc::DOMElement& s (
      ::xsd::cxx::xml::dom::create_element (
        "name",
        e));

    s << i.name ();
  }

  // rank
  //
  {
    ::xercesc::DOMElement& s (
      ::xsd::cxx::xml::dom::create_element (
        "rank",
        e));

    s << i.rank ();
  }

  // serial
  //
  {
    ::xercesc::DOMElement& s (
      ::xsd::cxx::xml::dom::create_element (
        "serial",
        e));

    s << i.serial ();
  }

  // loyalty
  //
  {
    ::xercesc::DOMAttr& a (
      ::xsd::cxx::xml::dom::create_attribute (
        "loyalty",
        e));

    a << i.loyalty ();
  }
}

#include <xsd/cxx/post.hxx>

// Begin epilogue.
//
//
// End epilogue.

